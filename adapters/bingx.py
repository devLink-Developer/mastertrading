"""
BingX perpetual futures adapter using CCXT.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import ccxt
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

_retry_exchange = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
    retry=retry_if_exception_type((ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout)),
    reraise=True,
)


class BingXFuturesAdapter:
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        password: str = "",
        sandbox: bool = False,
        margin_mode: str = "cross",
        leverage: int = 3,
    ):
        self.sandbox = sandbox
        self.margin_mode = margin_mode
        self.leverage = leverage
        self.client = ccxt.bingx(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "password": password or None,
                "enableRateLimit": True,
                "options": {"defaultType": "swap"},
            }
        )
        self._markets_loaded = False
        self._leverage_set_symbols: set[str] = set()
        if sandbox:
            try:
                self.client.set_sandbox_mode(True)
            except Exception as exc:
                logger.warning("set_sandbox_mode not available on bingx: %s", exc)
        self._ensure_markets_loaded()

    @classmethod
    def from_config(cls, cfg: dict | None = None) -> "BingXFuturesAdapter":
        if cfg is None:
            from .credentials import get_exchange_credentials

            cfg = get_exchange_credentials("bingx")
        key = str(cfg.get("api_key", "") or "")
        secret = str(cfg.get("api_secret", "") or "")
        password = str(cfg.get("api_passphrase", "") or "")
        sandbox = bool(cfg.get("sandbox", False))
        margin_mode = str(cfg.get("margin_mode", "cross") or "cross")
        leverage = int(cfg.get("leverage", 3) or 3)
        if not (key and secret):
            logger.warning("BINGX API creds missing; adapter will run public-only methods.")
        return cls(
            api_key=key,
            api_secret=secret,
            password=password,
            sandbox=sandbox,
            margin_mode=margin_mode,
            leverage=leverage,
        )

    def _ensure_markets_loaded(self):
        if self._markets_loaded:
            return
        try:
            self.client.load_markets()
            self._markets_loaded = True
        except Exception as exc:
            logger.warning("load_markets failed for bingx swap: %s", exc)

    def _map_symbol(self, symbol: str) -> str:
        if "/" in symbol:
            return symbol
        if symbol.endswith("USDT"):
            base = symbol[:-4]
            return f"{base}/USDT:USDT"
        return symbol

    @_retry_exchange
    def fetch_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 200, since: int | None = None):
        return self.client.fetch_ohlcv(self._map_symbol(symbol), timeframe=timeframe, limit=limit, since=since)

    @_retry_exchange
    def fetch_funding_rate_current(self, symbol: str):
        mapped = self._map_symbol(symbol)
        if hasattr(self.client, "fetch_funding_rate"):
            return self.client.fetch_funding_rate(mapped)
        return None

    @_retry_exchange
    def fetch_funding_rate_history(self, symbol: str, since: Optional[int] = None, limit: int = 50):
        mapped = self._map_symbol(symbol)
        if hasattr(self.client, "fetch_funding_rate_history"):
            return self.client.fetch_funding_rate_history(mapped, since=since, limit=limit)
        return []

    @_retry_exchange
    def fetch_ticker(self, symbol: str):
        return self.client.fetch_ticker(self._map_symbol(symbol))

    @_retry_exchange
    def fetch_balance(self):
        return self.client.fetch_balance()

    @_retry_exchange
    def _create_order_with_retry(
        self,
        mapped_symbol: str,
        side: str,
        type_: str,
        amount: float,
        price: Optional[float],
        params: Dict[str, Any],
    ):
        return self.client.create_order(mapped_symbol, type_, side, amount, price, params)

    def create_order(
        self,
        symbol: str,
        side: str,
        type_: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        params = dict(params or {})
        mapped = self._map_symbol(symbol)
        self._ensure_markets_loaded()
        try:
            if self.leverage and mapped not in self._leverage_set_symbols:
                # BingX requires explicit side when setting leverage.
                try:
                    self.client.set_leverage(self.leverage, mapped, {"side": "BOTH"})
                except Exception:
                    # Hedge-mode accounts may only accept LONG/SHORT.
                    self.client.set_leverage(self.leverage, mapped, {"side": "LONG"})
                    self.client.set_leverage(self.leverage, mapped, {"side": "SHORT"})
                self._leverage_set_symbols.add(mapped)
        except Exception as exc:
            logger.warning("set_leverage failed for %s: %s", mapped, exc)
        try:
            return self._create_order_with_retry(mapped, side, type_, amount, price, params)
        except Exception as exc:
            msg = str(exc).lower()
            hedge_requires_side = "positionside" in msg and "long or short" in msg
            hedge_rejects_reduce_only = "reduceonly" in msg and ("can not be filled" in msg or "cannot be filled" in msg)
            trigger_keys = ("triggerPrice", "stopPrice", "stopLossPrice", "takeProfitPrice", "stopLoss", "takeProfit")
            is_trigger_order = any(k in params for k in trigger_keys)
            if not (hedge_requires_side or hedge_rejects_reduce_only):
                raise

            retry_params = dict(params)
            if "positionSide" not in retry_params and "position_side" not in retry_params:
                if bool(retry_params.get("reduceOnly")):
                    # reduceOnly SELL usually closes LONG; reduceOnly BUY closes SHORT.
                    retry_params["positionSide"] = "LONG" if side.lower() == "sell" else "SHORT"
                else:
                    retry_params["positionSide"] = "LONG" if side.lower() == "buy" else "SHORT"

            logger.info(
                "Retrying BingX order (hedge mode) with positionSide=%s for %s",
                retry_params.get("positionSide"),
                mapped,
            )

            try:
                return self._create_order_with_retry(mapped, side, type_, amount, price, retry_params)
            except Exception as retry_exc:
                retry_msg = str(retry_exc).lower()
                reduce_only_rejected = "reduceonly" in retry_msg and ("can not be filled" in retry_msg or "cannot be filled" in retry_msg)
                if reduce_only_rejected and ("reduceOnly" in retry_params or "reduce_only" in retry_params):
                    hedged_params = dict(retry_params)
                    hedged_params.pop("reduceOnly", None)
                    hedged_params.pop("reduce_only", None)
                    # In BingX hedge mode, reduceOnly can be rejected even for valid closes.
                    # Keep close intent via explicit positionSide and hedged flag.
                    hedged_params["hedged"] = True
                    if "positionSide" not in hedged_params and "position_side" not in hedged_params:
                        hedged_params["positionSide"] = "LONG" if side.lower() == "sell" else "SHORT"
                    logger.info(
                        "Retrying BingX %s without reduceOnly using positionSide=%s for %s",
                        "trigger order" if is_trigger_order else "order",
                        hedged_params.get("positionSide"),
                        mapped,
                    )
                    try:
                        return self._create_order_with_retry(mapped, side, type_, amount, price, hedged_params)
                    except Exception as hedged_exc:
                        hedged_msg = str(hedged_exc).lower()
                        if (
                            "no position" in hedged_msg
                            or "position does not exist" in hedged_msg
                            or "position not exist" in hedged_msg
                            or "position size is 0" in hedged_msg
                        ):
                            raise ccxt.InvalidOrder(f"No position to close for {mapped}: {hedged_exc}")
                        raise ccxt.InvalidOrder(
                            f"Close order rejected for {mapped} after hedged retry ({hedged_exc})"
                        )
                raise

    @_retry_exchange
    def cancel_order(self, order_id: str, symbol: str):
        return self.client.cancel_order(order_id, self._map_symbol(symbol))

    @_retry_exchange
    def fetch_open_orders(self, symbol: Optional[str] = None):
        mapped = self._map_symbol(symbol) if symbol else None
        return self.client.fetch_open_orders(mapped)

    @_retry_exchange
    def fetch_open_stop_orders(self, symbol: Optional[str] = None):
        return self.fetch_open_orders(symbol)

    @_retry_exchange
    def fetch_positions(self, symbols: Optional[List[str]] = None):
        mapped = [self._map_symbol(s) for s in symbols] if symbols else None
        if hasattr(self.client, "fetch_positions"):
            return self.client.fetch_positions(symbols=mapped)
        return []
