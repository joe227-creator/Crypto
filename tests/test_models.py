from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_research import models


def test_ridge_embargo_is_previous_session_not_calendar_day(monkeypatch) -> None:
    dates = pd.bdate_range("2024-01-01", periods=6)
    features = pd.DataFrame(
        {
            "date": dates,
            "asset": "BTC",
            "ret_1": range(len(dates)),
            "label_end_date": dates,
            "label": np.arange(len(dates), dtype=float),
        }
    )
    captured: list[list[float]] = []

    def fake_fit(x: np.ndarray, y: np.ndarray, alpha: float):
        captured.append(x[:, 0].tolist())
        return np.zeros(2), np.zeros(1), np.ones(1)

    monkeypatch.setattr(models, "_fit_ridge", fake_fit)
    monkeypatch.setattr(models, "_predict", lambda model, x: np.zeros(len(x)))
    config = {
        "data": {"assets": ["BTC"]},
        "evaluation": {
            "train_start": "2024-01-01",
            "min_train_days": 3,
            "refit_sessions": 1,
            "embargo_sessions": 1,
        },
    }

    models.walk_forward_ridge(features, ["ret_1"], pd.DatetimeIndex([dates[-1]]), config, alpha=1.0, use_onchain=False)

    # Friday is previous session before Monday; its label must be purged.
    assert captured == [[0.0, 1.0, 2.0, 3.0]]
