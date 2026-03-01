from django.test import SimpleTestCase

from signals.meta_allocator import ModuleMetrics, compute_meta_weights_from_metrics


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
