"""
Nexubot Strategy Data Models.
Defines strict schemas for all SMC concepts (Swings, Events, Zones, Signals).
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, List, Optional


@dataclass
class SwingPoint:
    timestamp: datetime
    price: float
    type: Literal["high", "low"]
    classification: Literal["HH", "HL", "LH", "LL", "UNCONFIRMED"]
    candle_index: int


@dataclass
class StructureEvent:
    symbol: str
    timeframe: str
    event_type: Literal["BOS", "CHoCH", "MSS", "CISD"]
    direction: Literal["bullish", "bearish"]
    price_level: float
    timestamp: datetime
    confirmed: bool


@dataclass
class OrderBlock:
    id: str  # UUID for database
    symbol: str
    timeframe: str
    direction: Literal["bullish", "bearish"]
    ob_high: float
    ob_low: float
    ob_50: float  # Mitigation line
    strength_score: float  # 0.0 to 1.0
    mitigated: bool
    origin_timestamp: datetime
    mitigation_timestamp: Optional[datetime]


@dataclass
class FVG:
    symbol: str
    timeframe: str
    direction: Literal["bullish", "bearish"]
    top: float
    bottom: float
    timestamp: datetime
    mitigated: bool


@dataclass
class LiquidityPool:
    symbol: str
    timeframe: str
    pool_type: Literal["EQH", "EQL"]
    price_level: float
    swept: bool
    price_tolerance: float
    touch_count: int
    sweep_timestamp: Optional[datetime]


@dataclass
class OTEZone:
    symbol: str
    timeframe: str
    direction: Literal["bullish", "bearish"]
    fib_0: float
    fib_1: float
    ote_entry: float  # 61.8%
    ote_top: float  # 76.6%
    ote_mid: float  # 70.5%


@dataclass
class TradeSignal:
    signal_id: str
    symbol: str
    direction: Literal["buy", "sell"]
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    risk_reward: float
    signal_type: str
    entry_model: str
    session: str
    pd_zone: str
    confluence_score: int
    confluence_factors: List[str]
    timestamp: datetime
    timeframe: str
