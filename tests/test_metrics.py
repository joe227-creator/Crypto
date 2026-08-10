from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_research.metrics import complete_window_returns, research_score


def test_complete_windows_drop_incomplete_tail() -> None:
    index = pd.date_range("2018-01-01", periods=127, freq="D")
    equity = pd.Series(np.arange(1.0, 128.0), index=index)
    windows = complete_window_returns(equity, window_sessions=126)
    assert len(windows) == 1
    assert windows.iloc[0]["sessions"] == 126


def test_research_score_formula_is_fixed() -> None:
    result = research_score(0.10, -0.40, 1.0, 0.20, 0.10, 0.60)
    expected = 0.10 + 0.20 * 0.10 / 0.40 + 0.10 * 0.60 - 0.10 * (0.20 - 0.10)
    assert result["research_score"] == expected


def test_cash_score_is_finite() -> None:
    result = research_score(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    assert np.isfinite(result["research_score"])
