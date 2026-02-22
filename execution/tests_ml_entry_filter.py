from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from django.test import SimpleTestCase

from execution.ml_entry_filter import (
    FEATURE_NAMES,
    build_entry_feature_map,
    fit_logistic_model,
    load_model,
    predict_proba_from_model,
    save_model,
    vectorize_feature_map,
)


class EntryFilterMLHelpersTest(SimpleTestCase):
    def test_build_entry_feature_map_alloc_payload(self):
        payload = {
            "direction": "long",
            "confidence": 0.83,
            "raw_score": 0.77,
            "net_score": 0.42,
            "risk_budget_pct": 0.0027,
            "reasons": {
                "session": "asia",
                "active_module_count": 2,
                "module_rows": [{"module": "trend"}, {"module": "carry"}],
            },
        }
        fmap = build_entry_feature_map(
            strategy_name="alloc_long",
            symbol="BTCUSDT",
            sig_score=0.74,
            payload=payload,
            atr_pct=0.0019,
            spread_bps=8.2,
        )
        self.assertEqual(fmap["is_alloc"], 1.0)
        self.assertEqual(fmap["direction_long"], 1.0)
        self.assertEqual(fmap["session_asia"], 1.0)
        self.assertEqual(fmap["symbol_is_btc"], 1.0)
        self.assertEqual(fmap["active_module_count"], 2.0)
        self.assertAlmostEqual(fmap["atr_pct"], 0.0019, places=8)
        self.assertAlmostEqual(fmap["spread_bps"], 8.2, places=8)

    def test_fit_predict_and_persist_model(self):
        rows = []
        y = []
        for score, rb, target in [
            (0.92, 0.0040, 1),
            (0.88, 0.0032, 1),
            (0.84, 0.0026, 1),
            (0.31, 0.0002, 0),
            (0.27, 0.0001, 0),
            (0.20, 0.0000, 0),
            (0.74, 0.0020, 1),
            (0.41, 0.0007, 0),
            (0.66, 0.0018, 1),
            (0.35, 0.0003, 0),
            (0.77, 0.0019, 1),
            (0.38, 0.0004, 0),
        ]:
            fmap = build_entry_feature_map(
                strategy_name="alloc_long",
                symbol="ADAUSDT",
                sig_score=score,
                payload={
                    "direction": "long",
                    "confidence": score,
                    "raw_score": score,
                    "risk_budget_pct": rb,
                    "reasons": {"session": "asia", "active_module_count": 1},
                },
            )
            rows.append(vectorize_feature_map(fmap, feature_names=FEATURE_NAMES))
            y.append(float(target))

        model = fit_logistic_model(np.vstack(rows), np.asarray(y, dtype=np.float64), feature_names=FEATURE_NAMES)

        high = build_entry_feature_map(
            strategy_name="alloc_long",
            symbol="ADAUSDT",
            sig_score=0.91,
            payload={"direction": "long", "confidence": 0.9, "risk_budget_pct": 0.0035},
        )
        low = build_entry_feature_map(
            strategy_name="alloc_long",
            symbol="ADAUSDT",
            sig_score=0.24,
            payload={"direction": "long", "confidence": 0.2, "risk_budget_pct": 0.0001},
        )

        high_prob = predict_proba_from_model(model, high)
        low_prob = predict_proba_from_model(model, low)
        self.assertGreater(high_prob, low_prob)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "entry_model.json"
            save_model(model, path)
            loaded = load_model(path)
            loaded_high = predict_proba_from_model(loaded, high)
            self.assertAlmostEqual(high_prob, loaded_high, places=8)
