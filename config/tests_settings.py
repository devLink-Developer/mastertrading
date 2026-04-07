from __future__ import annotations

from django.test import SimpleTestCase

from config import settings as project_settings


class SettingsParsingTest(SimpleTestCase):
    def test_parse_raw_mapping_supports_relaxed_syntax(self):
        parsed = project_settings._parse_raw_mapping(
            "{BTCUSDT:base,ETHUSDT:base,SOLUSDT:mid,XRPUSDT:alt}"
        )
        self.assertEqual(
            parsed,
            {
                "BTCUSDT": "base",
                "ETHUSDT": "base",
                "SOLUSDT": "mid",
                "XRPUSDT": "alt",
            },
        )

    def test_instrument_tier_map_normalization_from_relaxed_mapping(self):
        parsed = {
            str(k).upper(): str(v).strip().lower()
            for k, v in project_settings._parse_raw_mapping(
                "{btcusdt:BASE,xrpusdt:Alt}"
            ).items()
            if str(k).strip() and str(v).strip()
        }
        self.assertEqual(parsed, {"BTCUSDT": "base", "XRPUSDT": "alt"})
