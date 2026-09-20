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
async def test_scan_ltf_entry_success(
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
    # Position final close inside active OB (2495-2505) and within discount range
    test_df.loc[test_df.index[-1], "close"] = 2500.0

    mock_get_candles.return_value = test_df
    mock_get_bias.return_value = "bullish"  # HTF agrees
    mock_get_active_obs.return_value = [dummy_ob_dict]
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
    assert "HTF Bias: BULLISH" in signal.confluence_factors

    mock_save_signal.assert_called_once()
    mock_save_pool.assert_called_once()


@pytest.mark.asyncio
@patch("strategies.confluence.candle_store.get_candles", new_callable=AsyncMock)
@patch("strategies.confluence.structure_events.get_latest_bias", new_callable=AsyncMock)
async def test_scan_ltf_entry_rejected_premium_discount(mock_get_bias, mock_get_candles, bullish_bos_df):
    """Test signal rejection when price is in Premium for a bullish setup."""
    test_df = bullish_bos_df.copy()
    # Place price at the top extreme of the range (Premium)
    test_df.loc[test_df.index[-1], "close"] = 2650.0

    mock_get_candles.return_value = test_df
    mock_get_bias.return_value = "bullish"

    engine = ConfluenceEngine(settings.SYMBOLS[0])
    signal = await engine.scan_ltf_entry()

    assert signal is None


@pytest.mark.asyncio
@patch("strategies.confluence.candle_store.get_candles", new_callable=AsyncMock)
@patch("strategies.confluence.structure_events.get_latest_bias", new_callable=AsyncMock)
@patch("strategies.confluence.order_blocks.get_active_order_blocks", new_callable=AsyncMock)
@patch("strategies.confluence.SessionManager.get_active_killzone")
async def test_scan_ltf_entry_low_confluence(
    mock_killzone, mock_get_active_obs, mock_get_bias, mock_get_candles, dummy_ob_dict, bullish_bos_df
):
    """Test signal rejection when confluence score is below threshold."""
    test_df = bullish_bos_df.copy()
    test_df.loc[test_df.index[-1], "close"] = 2500.0

    mock_get_candles.return_value = test_df
    mock_get_bias.return_value = "bullish"
    mock_get_active_obs.return_value = [dummy_ob_dict]
    mock_killzone.return_value = "Out of Session"

    # Only HTF bias (+20) and OB tap (+25) -> Score 45 < 70
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
async def test_scan_ltf_entry_breaker_block_fallback(
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

    mock_get_candles.return_value = test_df
    mock_get_bias.return_value = "bullish"
    mock_get_active_obs.return_value = []  # No standard active OB
    mock_get_breakers.return_value = [dummy_breaker_dict]  # Breaker available
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
    assert signal.entry_model == "Breaker Block Retest"
    assert "MTF Breaker Block Tap" in signal.confluence_factors

    mock_save_pool.assert_called_once()
    mock_save_signal.assert_called_once()


@pytest.mark.asyncio
@patch("strategies.confluence.candle_store.get_candles", new_callable=AsyncMock)
@patch("strategies.confluence.structure_events.get_latest_bias", new_callable=AsyncMock)
@patch("strategies.confluence.structure_events.save_event", new_callable=AsyncMock)
@patch("strategies.confluence.order_blocks.save_order_block", new_callable=AsyncMock)
async def test_scan_htf_and_mtf_scans(mock_save_ob, mock_save_event, mock_get_bias, mock_get_candles, bullish_bos_df):
    """Test HTF bias updates and MTF confirmation sweeps."""
    # Slice the DF to index 35 so the final closed candle evaluates as the exact breakout origin
    mock_get_candles.return_value = bullish_bos_df.iloc[:35].copy()
    mock_get_bias.return_value = "bullish"

    engine = ConfluenceEngine(settings.SYMBOLS[0])

    await engine.scan_htf()
    await engine.scan_mtf_confirmation()

    # BOS from bullish_bos_df should trigger structure_events persistence
    assert mock_save_event.call_count >= 1
