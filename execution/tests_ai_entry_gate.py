from django.test import SimpleTestCase

from execution.ai_entry_gate import _extract_json_blob, _parse_ai_decision


class AiEntryGateParseTest(SimpleTestCase):
    def test_extract_json_blob_direct(self):
        raw = '{"allow":false,"risk_mult":0.6,"reason":"weak"}'
        parsed = _extract_json_blob(raw)
        self.assertIsInstance(parsed, dict)
        assert parsed is not None
        self.assertEqual(parsed.get("allow"), False)

    def test_extract_json_blob_with_extra_text(self):
        raw = "output:\n```json\n{\"allow\": true, \"risk_mult\": 1.0, \"reason\": \"ok\"}\n```"
        parsed = _extract_json_blob(raw)
        self.assertIsInstance(parsed, dict)
        assert parsed is not None
        self.assertEqual(parsed.get("reason"), "ok")

    def test_parse_ai_decision_defaults_on_invalid(self):
        allow, risk_mult, reason = _parse_ai_decision("not-json")
        self.assertTrue(allow)
        self.assertEqual(risk_mult, 1.0)
        self.assertEqual(reason, "ai_parse_failed")

    def test_parse_ai_decision_clamps_risk(self):
        allow, risk_mult, reason = _parse_ai_decision('{"allow": true, "risk_mult": 1.8, "reason":"x"}')
        self.assertTrue(allow)
        self.assertEqual(risk_mult, 1.0)
        self.assertEqual(reason, "x")
