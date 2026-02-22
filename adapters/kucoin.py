"""
Kucoin Futures adapter using CCXT.
Supports sandbox via KUCOIN_SANDBOX env (default true).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import ccxt
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

# Retry decorator for transient exchange / network errors
_retry_exchange = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
    retry=retry_if_exception_type((ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout)),
    reraise=True,
)


class KucoinFuturesAdapter:
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        password: str,
        sandbox: bool = True,
        margin_mode: str = "cross",
        leverage: int = 3,
    ):
        self.sandbox = sandbox
        self.margin_mode = margin_mode
        self.leverage = leverage
        self.client = ccxt.kucoinfutures(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "password": password,
                "enableRateLimit": True,
            }
        )
        self._markets_loaded = False
        self._leverage_set_symbols: set[str] = set()
        if sandbox:
            self.client.set_sandbox_mode(True)
        self._ensure_markets_loaded()

    def _ensure_markets_loaded(self):
        if self._markets_loaded:
            return
        try:
            self.client.load_markets()
            self._markets_loaded = True
        except Exception as exc:
            logger.warning("load_markets failed for kucoin futures: %s", exc)

    def _map_symbol(self, symbol: str) -> str:
        # Accept formats like BTCUSDT or BTC/USDT:USDT
        if "/" in symbol:
            return symbol
        # KuCoin futures contract codes end with "USDTM" (e.g., XBTUSDTM)
        if symbol.endswith("USDTM"):
            base = symbol[:-5].replace("XBT", "BTC")  # strip 'USDTM'
            return f"{base}/USDT:USDT"
        if symbol.endswith("USDT"):
            base = symbol[:-4].replace("XBT", "BTC")
            return f"{base}/USDT:USDT"
        return symbol

    @classmethod
    def from_config(cls, cfg: dict | None = None) -> "KucoinFuturesAdapter":
        if cfg is None:
            from .credentials import get_exchange_credentials

            cfg = get_exchange_credentials("kucoin")
        key = str(cfg.get("api_key", "") or "")
        secret = str(cfg.get("api_secret", "") or "")
        password = str(cfg.get("api_passphrase", "") or "")
        sandbox = bool(cfg.get("sandbox", True))
        margin_mode = str(cfg.get("margin_mode", "cross") or "cross")
        leverage = int(cfg.get("leverage", 3) or 3)
        if not (key and secret and password):
            logger.warning("KUCOIN API creds missing; adapter will run public-only methods.")
        return cls(key, secret, password, sandbox=sandbox, margin_mode=margin_mode, leverage=leverage)

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

    def create_order(
        self,
        symbol: str,
        side: str,
        type_: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        params = params or {}
        params.setdefault("marginMode", self.margin_mode)
        if self.leverage:
            params.setdefault("leverage", self.leverage)
        mapped = self._map_symbol(symbol)
        self._ensure_markets_loaded()
        # Ensure leverage/margin mode are set before placing the order
        try:
            if self.leverage and mapped not in self._leverage_set_symbols:
                self.client.set_leverage(self.leverage, mapped, {"marginMode": self.margin_mode})
                self._leverage_set_symbols.add(mapped)
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("set_leverage failed for %s: %s", mapped, exc)
        return self.client.create_order(mapped, type_, side, amount, price, params)

    @_retry_exchange
    def cancel_order(self, order_id: str, symbol: str):
        return self.client.cancel_order(order_id, self._map_symbol(symbol))

    @_retry_exchange
    def fetch_open_orders(self, symbol: Optional[str] = None):
        mapped = self._map_symbol(symbol) if symbol else None
        return self.client.fetch_open_orders(mapped)

    @_retry_exchange
    def fetch_open_stop_orders(self, symbol: Optional[str] = None):
        """Fetch untriggered stop/trigger orders (separate endpoint on KuCoin Futures)."""
        mapped = self._map_symbol(symbol) if symbol else None
        return self.client.fetch_open_orders(mapped, params={"stop": True})

    @_retry_exchange
    def fetch_positions(self, symbols: Optional[List[str]] = None):
        mapped = [self._map_symbol(s) for s in symbols] if symbols else None
        if hasattr(self.client, "fetch_positions"):
            return self.client.fetch_positions(symbols=mapped)
        return []

    @_retry_exchange
    def fetch_closed_orders(self, symbol: str, since: Optional[int] = None, limit: int = 20):
        """Fetch recently closed/filled orders for a symbol."""
        mapped = self._map_symbol(symbol)
        return self.client.fetch_closed_orders(mapped, since=since, limit=limit)
