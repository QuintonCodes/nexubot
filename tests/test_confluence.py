"""
Unit tests for the Confluence Engine.
Run with: python -m pytest tests/test_confluence.py -v
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from config.settings import settings
from strategies.confluence import ConfluenceEngine
from strategies.models import LiquidityPool, StructureEvent


@pytest.mark.asyncio
@patch("strategies.confluence.candle_store.get_candles", new_callable=AsyncMock)
@patch("strategies.confluence.structure_events.get_latest_bias", new_callable=AsyncMock)
@patch("strategies.confluence.order_blocks.get_active_order_blocks", new_callable=AsyncMock)
@patch("strategies.confluence.signals.is_duplicate", new_callable=AsyncMock)
@patch("strategies.confluence.signals.save_signal", new_callable=AsyncMock)
@patch("strategies.confluence.liquidity_pools.save_pool", new_callable=AsyncMock)
@patch("strategies.confluence.detect_liquidity_sweep")
@patch("strategies.confluence.detect_choch")
@patch("strategies.confluence.SessionManager.get_active_killzone")
@patch("strategies.confluence.order_blocks.get_opposing_htf_obs", new_callable=AsyncMock)
@patch("strategies.confluence.liquidity_pools.get_active_pools", new_callable=AsyncMock)
async def test_scan_ltf_entry_success(
    mock_get_active_pools,
    mock_get_opposing_obs,
    mock_killzone,
    mock_detect_choch,
    mock_detect_sweep,
    mock_save_pool,
    mock_save_signal,
    mock_is_duplicate,
    mock_get_active_obs,
    mock_get_bias,
    mock_get_candles,
    dummy_ob_dict,
    bullish_bos_df,
):
    """Test that a signal is generated when price enters an active, HTF-aligned OB with high confluence."""
    test_df = bullish_bos_df.copy()
    test_df.loc[test_df.index[-1], "close"] = 2500.0

    mock_get_candles.return_value = test_df
    mock_get_bias.return_value = "bullish"  # HTF agrees
    mock_get_active_obs.return_value = [dummy_ob_dict]
    mock_get_opposing_obs.return_value = []
    mock_get_active_pools.return_value = []
    mock_is_duplicate.return_value = False
    mock_killzone.return_value = "London Open Killzone"

    # Mock sweeps and structural shift to reach minimum confluence threshold (>= 70)
    mock_detect_sweep.return_value = LiquidityPool(
        symbol="XAUUSD",
        timeframe="5min",
        pool_type="EQL",
        price_level=2495.0,
        price_tolerance=40.0,
        touch_count=2,
        swept=True,
        sweep_timestamp=datetime.now(timezone.utc),
    )
    mock_detect_choch.return_value = StructureEvent(
        event_type="CHoCH",
        direction="bullish",
        price_level=2500.0,
        timestamp=datetime.now(timezone.utc),
        symbol="XAUUSD",
        timeframe="5min",
        confirmed=True,
    )

    engine = ConfluenceEngine(settings.SYMBOLS[0])
    signal = await engine.scan_ltf_entry()

    assert signal is not None
    assert signal.direction == "buy"
    assert signal.entry_price == 2500.0
    assert signal.confluence_score >= settings.MIN_CONFLUENCE_SCORE
    assert "Full MTF/HTF Alignment (BULLISH)" in signal.confluence_factors

    mock_save_signal.assert_called_once()
    mock_save_pool.assert_called_once()


@pytest.mark.asyncio
@patch("strategies.confluence.candle_store.get_candles", new_callable=AsyncMock)
@patch("strategies.confluence.structure_events.get_latest_bias", new_callable=AsyncMock)
@patch("strategies.confluence.order_blocks.get_active_order_blocks", new_callable=AsyncMock)
@patch("strategies.confluence.order_blocks.get_active_breaker_blocks", new_callable=AsyncMock)
@patch("strategies.confluence.detect_liquidity_sweep")
@patch("strategies.confluence.order_blocks.get_opposing_htf_obs", new_callable=AsyncMock)
@patch("strategies.confluence.liquidity_pools.get_active_pools", new_callable=AsyncMock)
async def test_scan_ltf_entry_rejected_premium_discount(
    mock_get_active_pools,
    mock_get_opposing_obs,
    mock_detect_sweep,
    mock_get_breakers,
    mock_get_active_obs,
    mock_get_bias,
    mock_get_candles,
    bullish_bos_df,
):
    """Test signal rejection when price is in Premium for a bullish setup."""
    test_df = bullish_bos_df.copy()
    test_df.loc[test_df.index[-1], "close"] = 2650.0

    mock_get_candles.return_value = test_df
    mock_get_bias.return_value = "bullish"
    mock_get_active_obs.return_value = []
    mock_get_breakers.return_value = []
    mock_get_opposing_obs.return_value = []
    mock_get_active_pools.return_value = []
    mock_detect_sweep.return_value = None

    engine = ConfluenceEngine(settings.SYMBOLS[0])
    signal = await engine.scan_ltf_entry()

    assert signal is None


@pytest.mark.asyncio
@patch("strategies.confluence.candle_store.get_candles", new_callable=AsyncMock)
@patch("strategies.confluence.structure_events.get_latest_bias", new_callable=AsyncMock)
@patch("strategies.confluence.order_blocks.get_active_order_blocks", new_callable=AsyncMock)
@patch("strategies.confluence.signals.is_duplicate", new_callable=AsyncMock)
@patch("strategies.confluence.signals.save_signal", new_callable=AsyncMock)
@patch("strategies.confluence.liquidity_pools.save_pool", new_callable=AsyncMock)
@patch("strategies.confluence.detect_liquidity_sweep")
@patch("strategies.confluence.detect_choch")
@patch("strategies.confluence.SessionManager.get_active_killzone")
@patch("strategies.confluence.order_blocks.get_opposing_htf_obs", new_callable=AsyncMock)
@patch("strategies.confluence.liquidity_pools.get_active_pools", new_callable=AsyncMock)
async def test_scan_ltf_entry_accepted_premium_with_sweep(
    mock_get_active_pools,
    mock_get_opposing_obs,
    mock_killzone,
    mock_detect_choch,
    mock_detect_sweep,
    mock_save_pool,
    mock_save_signal,
    mock_is_duplicate,
    mock_get_active_obs,
    mock_get_bias,
    mock_get_candles,
    dummy_ob_dict,
    bullish_bos_df,
):
    """Test signal acceptance when price is in Premium for a bullish setup BUT there is a valid EQL sweep override."""
    test_df = bullish_bos_df.copy()
    test_df.loc[test_df.index[-1], "close"] = 2650.0  # Premium range

    dummy_ob_dict["ob_low"] = 2640.0
    dummy_ob_dict["ob_high"] = 2660.0

    mock_get_candles.return_value = test_df
    mock_get_bias.return_value = "bullish"
    mock_get_active_obs.return_value = [dummy_ob_dict]
    mock_get_opposing_obs.return_value = []
    mock_get_active_pools.return_value = []
    mock_is_duplicate.return_value = False
    mock_killzone.return_value = "London Open Killzone"

    mock_detect_sweep.return_value = LiquidityPool(
        symbol="XAUUSD",
        timeframe="5min",
        pool_type="EQL",
        price_level=2645.0,
        price_tolerance=40.0,
        touch_count=2,
        swept=True,
        sweep_timestamp=datetime.now(timezone.utc),
    )

    mock_detect_choch.return_value = None

    engine = ConfluenceEngine(settings.SYMBOLS[0])
    signal = await engine.scan_ltf_entry()

    assert signal is not None
    assert "PD Array Override (Liquidity Sweep / OTE)" in signal.confluence_factors
    mock_save_signal.assert_called_once()


@pytest.mark.asyncio
@patch("strategies.confluence.candle_store.get_candles", new_callable=AsyncMock)
@patch("strategies.confluence.structure_events.get_latest_bias", new_callable=AsyncMock)
@patch("strategies.confluence.order_blocks.get_active_order_blocks", new_callable=AsyncMock)
@patch("strategies.confluence.signals.is_duplicate", new_callable=AsyncMock)
@patch("strategies.confluence.SessionManager.get_active_killzone")
@patch("strategies.confluence.order_blocks.get_opposing_htf_obs", new_callable=AsyncMock)
@patch("strategies.confluence.liquidity_pools.get_active_pools", new_callable=AsyncMock)
async def test_scan_ltf_entry_low_confluence(
    mock_get_active_pools,
    mock_get_opposing_obs,
    mock_killzone,
    mock_is_duplicate,
    mock_get_active_obs,
    mock_get_bias,
    mock_get_candles,
    dummy_ob_dict,
    bullish_bos_df,
):
    """Test signal rejection when confluence score is below threshold."""
    test_df = bullish_bos_df.copy()
    test_df.loc[test_df.index[-1], "close"] = 2500.0

    mock_get_candles.return_value = test_df
    mock_get_active_obs.return_value = [dummy_ob_dict]
    mock_get_opposing_obs.return_value = []
    mock_get_active_pools.return_value = []

    def _mock_bias(symbol, tf):
        if tf == settings.HTF_TIMEFRAMES[0]:
            return "bullish"
        return "ranging"

    mock_get_bias.side_effect = _mock_bias
    mock_killzone.return_value = "Out of Session"

    engine = ConfluenceEngine(settings.SYMBOLS[0])
    signal = await engine.scan_ltf_entry()

    assert signal is None


@pytest.mark.asyncio
@patch("strategies.confluence.candle_store.get_candles", new_callable=AsyncMock)
@patch("strategies.confluence.structure_events.get_latest_bias", new_callable=AsyncMock)
@patch("strategies.confluence.order_blocks.get_active_order_blocks", new_callable=AsyncMock)
@patch("strategies.confluence.order_blocks.get_active_breaker_blocks", new_callable=AsyncMock)
@patch("strategies.confluence.signals.is_duplicate", new_callable=AsyncMock)
@patch("strategies.confluence.signals.save_signal", new_callable=AsyncMock)
@patch("strategies.confluence.liquidity_pools.save_pool", new_callable=AsyncMock)
@patch("strategies.confluence.detect_liquidity_sweep")
@patch("strategies.confluence.detect_choch")
@patch("strategies.confluence.SessionManager.get_active_killzone")
@patch("strategies.confluence.order_blocks.get_opposing_htf_obs", new_callable=AsyncMock)
@patch("strategies.confluence.liquidity_pools.get_active_pools", new_callable=AsyncMock)
async def test_scan_ltf_entry_breaker_block_fallback(
    mock_get_active_pools,
    mock_get_opposing_obs,
    mock_killzone,
    mock_detect_choch,
    mock_detect_sweep,
    mock_save_pool,
    mock_save_signal,
    mock_is_duplicate,
    mock_get_breakers,
    mock_get_active_obs,
    mock_get_bias,
    mock_get_candles,
    dummy_breaker_dict,
    bullish_bos_df,
):
    """Test fallback to Breaker Block when no active OB is tapped."""
    test_df = bullish_bos_df.copy()
    test_df.loc[test_df.index[-1], "close"] = 2500.0

    # Breaker must match the active trade bias (bullish) to trigger.
    dummy_breaker_dict["direction"] = "bullish"
    dummy_breaker_dict["ob_low"] = 2490.0
    dummy_breaker_dict["ob_high"] = 2510.0

    mock_get_candles.return_value = test_df
    mock_get_bias.return_value = "bullish"
    mock_get_active_obs.return_value = []
    mock_get_breakers.return_value = [dummy_breaker_dict]
    mock_get_opposing_obs.return_value = []
    mock_get_active_pools.return_value = []
    mock_is_duplicate.return_value = False
    mock_killzone.return_value = "London Open Killzone"

    mock_detect_sweep.return_value = LiquidityPool(
        symbol="XAUUSD",
        timeframe="5min",
        pool_type="EQL",
        price_level=2495.0,
        price_tolerance=40.0,
        touch_count=2,
        swept=True,
        sweep_timestamp=datetime.now(timezone.utc),
    )
    mock_detect_choch.return_value = StructureEvent(
        event_type="CHoCH",
        direction="bullish",
        price_level=2500.0,
        timestamp=datetime.now(timezone.utc),
        symbol="XAUUSD",
        timeframe="5min",
        confirmed=True,
    )

    engine = ConfluenceEngine(settings.SYMBOLS[0])
    signal = await engine.scan_ltf_entry()

    assert signal is not None
    assert signal.entry_model == "1h Breaker Block Retest"
    assert "1h Breaker Block Tap" in signal.confluence_factors

    mock_save_pool.assert_called_once()
    mock_save_signal.assert_called_once()


@pytest.mark.asyncio
@patch("strategies.confluence.candle_store.get_candles", new_callable=AsyncMock)
@patch("strategies.confluence.structure_events.get_latest_bias", new_callable=AsyncMock)
@patch("strategies.confluence.structure_events.save_event", new_callable=AsyncMock)
@patch("strategies.confluence.order_blocks.save_order_block", new_callable=AsyncMock)
@patch("strategies.confluence.order_blocks.mark_as_breaker", new_callable=AsyncMock)
async def test_scan_htf_and_mtf_scans(
    mock_mark_breaker, mock_save_ob, mock_save_event, mock_get_bias, mock_get_candles, bullish_bos_df
):
    """Test HTF bias updates and MTF confirmation sweeps."""
    mock_get_candles.return_value = bullish_bos_df.iloc[:35].copy()
    mock_get_bias.return_value = "ranging"

    engine = ConfluenceEngine(settings.SYMBOLS[0])

    await engine.scan_htf()
    await engine.scan_mtf_confirmation()

    assert mock_save_event.call_count >= 1
