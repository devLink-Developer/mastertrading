from django.test import SimpleTestCase

from signals.meta_allocator import (
    ModuleMetrics,
    _risk_budgets_from_weights,
    compute_meta_weights_from_metrics,
)


class MetaAllocatorWeightsTest(SimpleTestCase):
    def test_compute_meta_weights_prefers_higher_expectancy(self):
        base = {"trend": 0.4, "meanrev": 0.3, "carry": 0.2, "smc": 0.1}
        metrics = {
            "trend": ModuleMetrics(n=40, expectancy=0.003, stdev=0.004, profit_factor=1.5, loss_cluster=0.10),
            "meanrev": ModuleMetrics(n=40, expectancy=0.001, stdev=0.004, profit_factor=1.1, loss_cluster=0.10),
            "carry": ModuleMetrics(n=40, expectancy=-0.001, stdev=0.004, profit_factor=0.8, loss_cluster=0.20),
            "smc": ModuleMetrics(n=40, expectancy=0.0005, stdev=0.006, profit_factor=1.0, loss_cluster=0.15),
        }
        w, diag = compute_meta_weights_from_metrics(
            base_weights=base,
            metrics=metrics,
            weight_cap=0.8,
            loss_cluster_penalty=0.5,
            pf_target=1.2,
            single_winner_enabled=False,
            single_winner_min_weight=0.5,
        )
        self.assertAlmostEqual(sum(w.values()), 1.0, places=6)
        self.assertGreater(w["trend"], w["meanrev"])
        self.assertGreater(w["meanrev"], w["carry"])
        self.assertIn("module_metrics", diag)

    def test_compute_meta_weights_respects_normalized_weight_cap(self):
        base = {"trend": 0.4, "meanrev": 0.2, "carry": 0.3, "grid": 0.0, "smc": 0.1}
        metrics = {
            "trend": ModuleMetrics(n=40, expectancy=-0.002, stdev=0.010, profit_factor=0.6, loss_cluster=0.35),
            "meanrev": ModuleMetrics(n=40, expectancy=-0.001, stdev=0.011, profit_factor=0.7, loss_cluster=0.30),
            "carry": ModuleMetrics(n=40, expectancy=0.008, stdev=0.003, profit_factor=2.5, loss_cluster=0.02),
            "smc": ModuleMetrics(n=40, expectancy=-0.002, stdev=0.012, profit_factor=0.5, loss_cluster=0.40),
        }
        w, _diag = compute_meta_weights_from_metrics(
            base_weights=base,
            metrics=metrics,
            weight_cap=0.55,
            loss_cluster_penalty=0.5,
            pf_target=1.2,
            single_winner_enabled=False,
            single_winner_min_weight=0.5,
        )
        self.assertAlmostEqual(sum(w.values()), 1.0, places=6)
        self.assertLessEqual(max(w.values()), 0.55 + 1e-6)
        self.assertGreater(w["carry"], w["trend"])

    def test_under_sampled_modules_keep_base_share_instead_of_zero(self):
        base = {"trend": 0.25, "meanrev": 0.20, "carry": 0.15, "grid": 0.15, "smc": 0.25}
        metrics = {
            "trend": ModuleMetrics(
                n=93,
                expectancy=-0.001185,
                stdev=0.005758,
                profit_factor=0.597196,
                loss_cluster=0.129032,
                corr_penalty=0.793259,
                data_readiness=1.0,
            ),
            "meanrev": ModuleMetrics(
                n=3,
                expectancy=0.0004,
                stdev=0.002,
                profit_factor=1.1,
                loss_cluster=0.0,
                corr_penalty=1.0,
                data_readiness=0.25,
            ),
            "carry": ModuleMetrics(
                n=83,
                expectancy=-0.00042,
                stdev=0.001265,
                profit_factor=0.366908,
                loss_cluster=0.13253,
                corr_penalty=0.727028,
                data_readiness=1.0,
            ),
            "grid": ModuleMetrics(n=0, corr_penalty=1.0, data_readiness=0.0),
            "smc": ModuleMetrics(
                n=9,
                expectancy=-0.0003,
                stdev=0.004,
                profit_factor=0.9,
                loss_cluster=0.10,
                corr_penalty=1.0,
                data_readiness=0.75,
            ),
        }
        w, diag = compute_meta_weights_from_metrics(
            base_weights=base,
            metrics=metrics,
            weight_cap=0.55,
            loss_cluster_penalty=0.5,
            pf_target=1.2,
            single_winner_enabled=False,
            single_winner_min_weight=0.5,
        )
        self.assertAlmostEqual(sum(w.values()), 1.0, places=6)
        self.assertGreater(w["meanrev"], 0.0)
        self.assertGreater(w["grid"], 0.0)
        self.assertGreater(w["smc"], 0.0)
        self.assertLessEqual(max(w.values()), 0.55 + 1e-6)
        self.assertEqual(diag["module_metrics"]["grid"]["reason"], "no_data_keep_base")
        self.assertAlmostEqual(diag["module_metrics"]["meanrev"]["data_readiness"], 0.25, places=6)

    def test_trend_floor_preserves_some_weight_during_strong_meta_penalty(self):
        base = {"trend": 0.215, "meanrev": 0.219, "carry": 0.128, "grid": 0.164, "smc": 0.274}
        metrics = {
            "trend": ModuleMetrics(
                n=93,
                expectancy=-0.001185,
                stdev=0.005758,
                profit_factor=0.597196,
                loss_cluster=0.129032,
                corr_penalty=0.793259,
                data_readiness=1.0,
            ),
            "meanrev": ModuleMetrics(
                n=3,
                expectancy=0.001459,
                stdev=0.003102,
                profit_factor=2.935539,
                loss_cluster=0.333333,
                corr_penalty=1.0,
                data_readiness=0.25,
            ),
            "carry": ModuleMetrics(
                n=83,
                expectancy=-0.00042,
                stdev=0.001265,
                profit_factor=0.366908,
                loss_cluster=0.13253,
                corr_penalty=0.727028,
                data_readiness=1.0,
            ),
            "grid": ModuleMetrics(n=0, corr_penalty=1.0, data_readiness=0.0),
            "smc": ModuleMetrics(
                n=9,
                expectancy=-0.003154,
                stdev=0.0038,
                profit_factor=0.159269,
                loss_cluster=0.777778,
                corr_penalty=0.85949,
                data_readiness=0.75,
            ),
        }
        w_no_floor, _ = compute_meta_weights_from_metrics(
            base_weights=base,
            metrics=metrics,
            weight_cap=0.55,
            loss_cluster_penalty=0.5,
            pf_target=1.2,
            single_winner_enabled=False,
            single_winner_min_weight=0.5,
        )
        w_with_floor, diag = compute_meta_weights_from_metrics(
            base_weights=base,
            metrics=metrics,
            weight_cap=0.55,
            loss_cluster_penalty=0.5,
            pf_target=1.2,
            single_winner_enabled=False,
            single_winner_min_weight=0.5,
            min_base_weight_share_by_module={"trend": 0.35},
        )
        self.assertAlmostEqual(sum(w_with_floor.values()), 1.0, places=6)
        self.assertGreater(w_with_floor["trend"], w_no_floor["trend"])
        self.assertGreater(w_with_floor["trend"], 0.10)
        self.assertAlmostEqual(diag["module_metrics"]["trend"]["min_base_share"], 0.35, places=6)
        self.assertGreater(diag["module_metrics"]["trend"]["raw_floor"], 0.0)

    def test_single_winner_mode_when_top_weight_is_high(self):
        base = {"trend": 0.6, "meanrev": 0.2, "carry": 0.1, "smc": 0.1}
        metrics = {
            "trend": ModuleMetrics(n=50, expectancy=0.004, stdev=0.003, profit_factor=1.8, loss_cluster=0.05),
            "meanrev": ModuleMetrics(n=50, expectancy=-0.002, stdev=0.008, profit_factor=0.7, loss_cluster=0.25),
            "carry": ModuleMetrics(n=50, expectancy=-0.001, stdev=0.007, profit_factor=0.8, loss_cluster=0.20),
            "smc": ModuleMetrics(n=50, expectancy=0.0001, stdev=0.010, profit_factor=0.9, loss_cluster=0.30),
        }
        w, diag = compute_meta_weights_from_metrics(
            base_weights=base,
            metrics=metrics,
            weight_cap=1.0,
            loss_cluster_penalty=0.5,
            pf_target=1.2,
            single_winner_enabled=True,
            single_winner_min_weight=0.40,
        )
        self.assertEqual(diag.get("winner"), "trend")
        self.assertEqual(w["trend"], 1.0)
        self.assertEqual(w["meanrev"], 0.0)
        self.assertEqual(w["carry"], 0.0)
        self.assertEqual(w["smc"], 0.0)

    def test_p4_freezes_bucket_when_strategy_guard_is_zero(self):
        base = {"trend": 0.4, "meanrev": 0.3, "carry": 0.2, "smc": 0.1}
        metrics = {
            "trend": ModuleMetrics(
                n=80,
                expectancy=0.004,
                stdev=0.003,
                profit_factor=1.6,
                loss_cluster=0.10,
                bucket_freeze=True,
                dd_throttle_mult=0.0,
            ),
            "meanrev": ModuleMetrics(
                n=80,
                expectancy=0.002,
                stdev=0.004,
                profit_factor=1.3,
                loss_cluster=0.08,
                bucket_freeze=False,
                dd_throttle_mult=1.0,
                daily_loss_throttle_mult=1.0,
            ),
        }
        w, diag = compute_meta_weights_from_metrics(
            base_weights=base,
            metrics=metrics,
            weight_cap=1.0,
            loss_cluster_penalty=0.5,
            pf_target=1.2,
            single_winner_enabled=False,
            single_winner_min_weight=0.5,
            p4_enabled=True,
            p4_min_sample=50,
        )
        self.assertAlmostEqual(w["trend"], 0.0, places=6)
        self.assertGreater(w["meanrev"], 0.95)
        self.assertTrue(bool(diag["module_metrics"]["trend"]["bucket_freeze"]))

    def test_strict_risk_budgets_do_not_auto_renormalize(self):
        budgets = _risk_budgets_from_weights(
            weights={"trend": 1.0, "meanrev": 0.0, "carry": 0.0, "smc": 0.0},
            fallback_budgets={"trend": 0.25, "meanrev": 0.25, "carry": 0.25, "smc": 0.25},
            bucket_caps={"trend": 0.20, "meanrev": 0.30, "carry": 0.40, "smc": 0.50},
            strict_isolation=True,
            max_total_budget=1.0,
        )
        self.assertAlmostEqual(sum(budgets.values()), 0.2, places=6)
        self.assertAlmostEqual(budgets["trend"], 0.2, places=6)
        self.assertAlmostEqual(budgets["meanrev"], 0.0, places=6)

    def test_non_strict_risk_budgets_renormalize(self):
        budgets = _risk_budgets_from_weights(
            weights={"trend": 1.0, "meanrev": 0.0, "carry": 0.0, "smc": 0.0},
            fallback_budgets={"trend": 0.25, "meanrev": 0.25, "carry": 0.25, "smc": 0.25},
            bucket_caps={"trend": 0.20, "meanrev": 0.30, "carry": 0.40, "smc": 0.50},
            strict_isolation=False,
            max_total_budget=1.0,
        )
        self.assertAlmostEqual(sum(budgets.values()), 1.0, places=6)
        self.assertAlmostEqual(budgets["trend"], 1.0, places=6)
