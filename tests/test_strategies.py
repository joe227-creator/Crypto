from __future__ import annotations

import pandas as pd

from crypto_research import strategies


def test_timesfm_confidence_mode_applies_uncertainty_gate(monkeypatch) -> None:
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    market = pd.concat(
        [
            pd.DataFrame(
                {
                    "date": dates,
                    "asset": asset,
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.0,
                    "close": 10.0,
                    "volume": 100.0,
                }
            )
            for asset in ["BTC", "ETH"]
        ],
        ignore_index=True,
    )
    features = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "asset": ["BTC"] * len(dates) + ["ETH"] * len(dates),
            "ret_1": 0.0,
        }
    )

    def fake_timesfm(*args, **kwargs):
        return pd.DataFrame(
            [
                {"date": date, "asset": asset, "prediction": 0.1, "uncertainty": 0.2}
                for date in dates
                for asset in ["BTC", "ETH"]
            ]
        )

    monkeypatch.setattr(strategies, "walk_forward_timesfm", fake_timesfm)
    targets, _ = strategies.build_targets(
        market,
        features,
        [],
        {"data": {"assets": ["BTC", "ETH"]}},
        mode="timesfm_confidence",
        params={"max_uncertainty": 0.1},
    )

    assert not targets.empty
    assert (targets[["BTC", "ETH"]] == 0.0).all().all()
