from django.test import SimpleTestCase

from core.models import ApiProviderConfig
from execution.ai_exit_gate import (
    _build_responses_body,
    _parse_ai_exit_decision,
)


class AiExitGateParseTest(SimpleTestCase):
    def test_parse_close_json(self):
        close_now, reason = _parse_ai_exit_decision('{"action":"close","reason":"reversal_risk"}')
        self.assertTrue(close_now)
        self.assertEqual(reason, "reversal_risk")

    def test_parse_hold_nested_json(self):
        close_now, reason = _parse_ai_exit_decision(
            '{"decision":{"action":"hold","reason":"momentum_ok"}}'
        )
        self.assertFalse(close_now)
        self.assertEqual(reason, "momentum_ok")

    def test_parse_heuristic_text(self):
        close_now, reason = _parse_ai_exit_decision("action=close; reason=tp_near_and_weakening")
        self.assertTrue(close_now)
        self.assertEqual(reason, "tp_near_and_weakening")

    def test_parse_invalid_falls_back_to_hold(self):
        close_now, reason = _parse_ai_exit_decision("not-json")
        self.assertFalse(close_now)
        self.assertEqual(reason, "ai_exit_parse_failed")


class AiExitGateRequestBodyTest(SimpleTestCase):
    def _cfg(self, model_name: str, extra_params: dict | None = None):
        cfg = ApiProviderConfig(
            name_alias="t",
            provider=ApiProviderConfig.Provider.OPENAI,
            model_name=model_name,
            temperature=0.2,
            top_p=1.0,
            max_output_tokens=120,
            max_input_tokens=12000,
            timeout_seconds=30,
            active=True,
        )
        cfg.extra_params_json = extra_params or {}
        return cfg

    def test_build_body_uses_json_schema_for_openai(self):
        cfg = self._cfg("gpt-5")
        body = _build_responses_body(
            cfg=cfg,
            system_msg="sys",
            user_msg="usr",
            reserve_out=96,
        )
        self.assertEqual((body.get("reasoning") or {}).get("effort"), "minimal")
        self.assertEqual((((body.get("text") or {}).get("format") or {}).get("type")), "json_schema")
        self.assertNotIn("temperature", body)
        self.assertNotIn("top_p", body)

    def test_build_body_respects_supported_sampling_controls(self):
        cfg = self._cfg("gpt-4o-mini")
        body = _build_responses_body(
            cfg=cfg,
            system_msg="sys",
            user_msg="usr",
            reserve_out=96,
        )
        self.assertEqual(body.get("temperature"), 0.2)
        self.assertEqual(body.get("top_p"), 1.0)
