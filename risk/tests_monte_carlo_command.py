from django.test import SimpleTestCase

from risk.management.commands.monte_carlo import Command


class MonteCarloCommandTest(SimpleTestCase):
    def test_resolve_stress_profile_defaults(self):
        cmd = Command()
        stress = cmd._resolve_stress_params({"stress_profile": "balanced"})
        self.assertEqual(stress["profile"], "balanced")
        self.assertGreaterEqual(stress["vol_mult"], 1.0)
        self.assertGreaterEqual(stress["loss_cluster_mult"], 1.0)

    def test_resolve_stress_profile_overrides(self):
        cmd = Command()
        stress = cmd._resolve_stress_params(
            {
                "stress_profile": "none",
                "stress_vol_mult": 1.9,
                "stress_loss_cluster_mult": 1.6,
                "stress_corr_shock_prob": 0.2,
                "stress_corr_shock_size": 0.01,
                "stress_bear_bias": 0.33,
            }
        )
        self.assertEqual(stress["profile"], "none")
        self.assertAlmostEqual(stress["vol_mult"], 1.9, places=6)
        self.assertAlmostEqual(stress["loss_cluster_mult"], 1.6, places=6)
        self.assertAlmostEqual(stress["corr_shock_prob"], 0.2, places=6)
        self.assertAlmostEqual(stress["corr_shock_size"], 0.01, places=6)
        self.assertAlmostEqual(stress["bear_bias"], 0.33, places=6)
