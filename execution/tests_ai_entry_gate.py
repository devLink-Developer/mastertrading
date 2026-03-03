from django.test import SimpleTestCase

from core.models import ApiProviderConfig
from execution.ai_entry_gate import (
    _build_gate_messages,
    _build_responses_body,
    _collect_output_text,
    _compact_payload,
    _dir_code,
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

    def test_parse_ai_decision_accepts_nested_decision_payload(self):
        allow, risk_mult, reason = _parse_ai_decision(
            '{"decision":{"allow":false,"risk_multiplier":0.42,"rationale":"regime_conflict"}}'
        )
        self.assertFalse(allow)
        self.assertEqual(risk_mult, 0.42)
        self.assertEqual(reason, "regime_conflict")

    def test_parse_ai_decision_accepts_python_dict_like_text(self):
        allow, risk_mult, reason = _parse_ai_decision("{'allow': True, 'risk_mult': 0.7, 'reason': 'ok'}")
        self.assertTrue(allow)
        self.assertEqual(risk_mult, 0.7)
        self.assertEqual(reason, "ok")

    def test_parse_ai_decision_uses_heuristic_fallback_when_not_json(self):
        allow, risk_mult, reason = _parse_ai_decision(
            "allow=no; risk_multiplier=0.35; reason=spread_too_wide"
        )
        self.assertFalse(allow)
        self.assertEqual(risk_mult, 0.35)
        self.assertEqual(reason, "spread_too_wide")


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
        self.assertEqual((body.get("reasoning") or {}).get("effort"), "minimal")
        self.assertEqual(
            (((body.get("text") or {}).get("format") or {}).get("type")),
            "json_schema",
        )

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

    def test_build_body_allows_overriding_reasoning_or_text_from_extra_params(self):
        cfg = self._cfg(
            "gpt-5",
            extra_params={
                "reasoning": {"effort": "low"},
                "text": {"format": {"type": "text"}},
            },
        )
        body = _build_responses_body(
            cfg=cfg,
            system_msg="sys",
            user_msg="usr",
            reserve_out=100,
        )
        self.assertEqual((body.get("reasoning") or {}).get("effort"), "low")
        self.assertEqual((((body.get("text") or {}).get("format") or {}).get("type")), "text")


class AiEntryGateCollectOutputText(SimpleTestCase):
    def test_collect_output_text_reads_message_after_reasoning(self):
        data = {
            "output_text": None,
            "output": [
                {"type": "reasoning", "summary": []},
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": '{"allow":true,"risk_mult":0.9,"reason":"ok"}',
                        }
                    ],
                },
            ],
        }
        self.assertEqual(_collect_output_text(data), '{"allow":true,"risk_mult":0.9,"reason":"ok"}')

    def test_collect_output_text_reads_structured_json_content(self):
        data = {
            "output_text": "",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_json",
                            "json": {"allow": False, "risk_mult": 0.4, "reason": "spread"},
                        }
                    ],
                }
            ],
        }
        txt = _collect_output_text(data)
        self.assertIn('"allow":false', txt)
        self.assertIn('"risk_mult":0.4', txt)


class AiEntryGatePromptCompactTest(SimpleTestCase):
    def test_dir_code(self):
        self.assertEqual(_dir_code("long"), "l")
        self.assertEqual(_dir_code("short"), "s")
        self.assertEqual(_dir_code("LONG"), "l")
        self.assertEqual(_dir_code(""), "u")

    def test_compact_payload_uses_short_keys(self):
        payload = {
            "reasons": {
                "net_score": 0.42,
                "module_rows": [
                    {
                        "module": "trend",
                        "direction": "long",
                        "confidence": 0.8,
                        "raw_score": 0.3,
                    }
                ],
            },
            "risk_budget_pct": 0.003,
            "entry_reason": "x",
            "regime": "trending",
            "session": "london",
        }
        out = _compact_payload(payload)
        self.assertEqual(out.get("ns"), 0.42)
        self.assertEqual(out.get("rb"), 0.003)
        self.assertEqual(out.get("er"), "x")
        self.assertEqual(out.get("rg"), "trending")
        self.assertEqual(out.get("se"), "london")
        self.assertEqual(out.get("mr"), [["trend", "l", 0.8, 0.3]])

    def test_build_gate_messages_omits_empty_sections(self):
        system_msg, user_msg = _build_gate_messages(
            user_prompt='{"sym":"BTCUSDT"}',
            ctx_text="",
            feedback_text="",
        )
        self.assertIn("Return JSON only", system_msg)
        self.assertEqual(user_msg, 'in={"sym":"BTCUSDT"}')

    def test_build_gate_messages_includes_ctx_feedback_when_present(self):
        _, user_msg = _build_gate_messages(
            user_prompt='{"sym":"BTCUSDT"}',
            ctx_text="ctx-line",
            feedback_text='{"ev":"x"}',
        )
        self.assertIn('in={"sym":"BTCUSDT"}', user_msg)
        self.assertIn("ctx=ctx-line", user_msg)
        self.assertIn('fb={"ev":"x"}', user_msg)
