from django.test import SimpleTestCase

from core.models import ApiProviderConfig
from execution.ai_entry_gate import (
    _build_responses_body,
    _extract_json_blob,
    _parse_ai_decision,
    _supports_sampling_controls,
)


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


class AiEntryGateRequestBodyTest(SimpleTestCase):
    def _cfg(self, model_name: str, extra_params: dict | None = None):
        cfg = ApiProviderConfig(
            name_alias="t",
            provider=ApiProviderConfig.Provider.OPENAI,
            model_name=model_name,
            temperature=0.2,
            top_p=1.0,
            max_output_tokens=180,
            max_input_tokens=12000,
            timeout_seconds=30,
            active=True,
        )
        cfg.extra_params_json = extra_params or {}
        return cfg

    def test_supports_sampling_controls_for_gpt4o(self):
        self.assertTrue(_supports_sampling_controls("gpt-4o"))
        self.assertTrue(_supports_sampling_controls("gpt-4o-mini"))

    def test_disables_sampling_controls_for_gpt5_family(self):
        self.assertFalse(_supports_sampling_controls("gpt-5"))
        self.assertFalse(_supports_sampling_controls("gpt-5-mini"))

    def test_build_body_includes_temperature_top_p_for_supported_models(self):
        cfg = self._cfg("gpt-4o")
        body = _build_responses_body(
            cfg=cfg,
            system_msg="sys",
            user_msg="usr",
            reserve_out=100,
        )
        self.assertIn("temperature", body)
        self.assertIn("top_p", body)
        self.assertEqual(body["temperature"], 0.2)
        self.assertEqual(body["top_p"], 1.0)

    def test_build_body_omits_temperature_top_p_for_gpt5_models(self):
        cfg = self._cfg("gpt-5")
        body = _build_responses_body(
            cfg=cfg,
            system_msg="sys",
            user_msg="usr",
            reserve_out=100,
        )
        self.assertNotIn("temperature", body)
        self.assertNotIn("top_p", body)

    def test_build_body_strips_sampling_controls_injected_via_extra_params(self):
        cfg = self._cfg(
            "gpt-5-mini",
            extra_params={"temperature": 0.9, "top_p": 0.2, "foo": "bar"},
        )
        body = _build_responses_body(
            cfg=cfg,
            system_msg="sys",
            user_msg="usr",
            reserve_out=100,
        )
        self.assertNotIn("temperature", body)
        self.assertNotIn("top_p", body)
        self.assertEqual(body.get("foo"), "bar")
