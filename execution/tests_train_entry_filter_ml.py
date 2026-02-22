from __future__ import annotations

import numpy as np
from django.test import SimpleTestCase

from execution.management.commands.train_entry_filter_ml import _choose_holdout_split


class TrainEntryFilterCommandHelpersTest(SimpleTestCase):
    def test_choose_holdout_split_requires_enough_samples(self):
        y = np.array([0, 1] * 30, dtype=np.float64)
        self.assertIsNone(_choose_holdout_split(y))

    def test_choose_holdout_split_returns_index_for_balanced_series(self):
        y = np.array([0, 1] * 60, dtype=np.float64)
        split_idx = _choose_holdout_split(y)
        self.assertIsNotNone(split_idx)
        self.assertGreater(split_idx, 0)
        self.assertLess(split_idx, y.size)
