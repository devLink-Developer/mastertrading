from __future__ import annotations

import os
from typing import Protocol

from .binance import BinanceFuturesAdapter
from .bingx import BingXFuturesAdapter
from .credentials import (
    get_active_service,
    get_default_adapter_signature,
    get_exchange_credentials,
)
from .kucoin import KucoinFuturesAdapter


class AdapterProtocol(Protocol):
    leverage: int
    margin_mode: str

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 200):
        ...

    def fetch_funding_rate_current(self, symbol: str):
        ...

    def fetch_funding_rate_history(self, symbol: str, since=None, limit: int = 50):
        ...

    def fetch_ticker(self, symbol: str):
        ...

    def fetch_balance(self):
        ...

    def create_order(
        self, symbol: str, side: str, type_: str, amount: float, price=None, params=None
    ):
        ...

    def cancel_order(self, order_id: str, symbol: str):
        ...

    def fetch_open_orders(self, symbol: str | None = None):
        ...

    def fetch_positions(self, symbols=None):
        ...

    def _map_symbol(self, symbol: str) -> str:
        ...


def get_default_adapter() -> AdapterProtocol:
    exchange = get_active_service(default_env=os.getenv("EXCHANGE", "kucoin"))
    cfg = get_exchange_credentials(exchange)
    if exchange == "bingx":
        return BingXFuturesAdapter.from_config(cfg)
    if exchange == "binance":
        return BinanceFuturesAdapter.from_config(cfg)
    return KucoinFuturesAdapter.from_config(cfg)
