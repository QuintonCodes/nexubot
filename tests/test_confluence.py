"""
Unit tests for the Confluence Engine (Phase 4).
Run with: python -m pytest tests/test_confluence.py -v
"""

import pytest
import uuid
from datetime import datetime, timezone
from unittest.mock import patch, AsyncMock

from config.settings import settings
from strategies.confluence import ConfluenceEngine
from strategies.models import OrderBlock


@pytest.fixture
def dummy_ob():
    return OrderBlock(
        id=str(uuid.uuid4()),
        symbol="XAU/USD",
        timeframe="1h",
        direction="bullish",
        ob_high=2505.0,
        ob_low=2495.0,
        ob_50=2500.0,
        origin_timestamp=datetime.now(timezone.utc),
        is_mitigated=False,
        mitigation_timestamp=None,
        strength_score=0.8,
    )


@pytest.mark.asyncio
@patch("strategies.confluence.candle_store.get_candles", new_callable=AsyncMock)
@patch("strategies.confluence.structure_events.get_latest_bias", new_callable=AsyncMock)
@patch("strategies.confluence.order_blocks.get_active_order_blocks", new_callable=AsyncMock)
@patch("strategies.confluence.signals.is_duplicate", new_callable=AsyncMock)
@patch("strategies.confluence.signals.save_signal", new_callable=AsyncMock)
async def test_scan_ltf_entry_success(
    mock_save_signal, mock_is_duplicate, mock_get_active_obs, mock_get_bias, mock_get_candles, dummy_ob, bullish_bos_df
):
    """Test that a signal is generated when price enters an active, HTF-aligned Order Block."""

    # 1. Setup mocks
    # Manipulate the last candle to be exactly inside our dummy_ob (2495 - 2505)
    test_df = bullish_bos_df.copy()
    test_df.loc[test_df.index[-1], "close"] = 2502.0

    mock_get_candles.return_value = test_df
    mock_get_bias.return_value = "bullish"  # HTF agrees
    mock_get_active_obs.return_value = [dummy_ob]  # There is an active MTF OB
    mock_is_duplicate.return_value = False  # Not a duplicate

    # 2. Execute
    engine = ConfluenceEngine(settings.SYMBOLS[0])
    signal = await engine.scan_ltf_entry()

    # 3. Assertions
    assert signal is not None
    assert signal.direction == "buy"
    assert signal.entry_price == 2502.0
    assert "HTF Bias: BULLISH" in signal.confluence_factors

    mock_save_signal.assert_called_once()
