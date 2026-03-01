from __future__ import annotations

import json
import shutil
from pathlib import Path

from django.conf import settings
from django.core.management import call_command
from django.test import TestCase, override_settings

from core.ai_feedback import load_feedback_context_tail, record_ai_feedback_event
from core.models import AiFeedbackEvent, ApiProviderConfig


class AiFeedbackTest(TestCase):
    def setUp(self):
        self.cfg = ApiProviderConfig.objects.create(
            name_alias="gpt-feedback",
            provider=ApiProviderConfig.Provider.OPENAI,
            model_name="gpt-5",
            active=True,
            is_default=True,
            api_key="dummy",
        )
        self.tmp_dir = Path(settings.BASE_DIR) / "tmp" / "tests_ai_feedback"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.rel_stream = (self.tmp_dir / "stream.jsonl").relative_to(settings.BASE_DIR).as_posix()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    @override_settings(AI_FEEDBACK_JSONL_ENABLED=True, AI_FEEDBACK_JSONL_MAX_BYTES=2_000_000)
    def test_record_feedback_persists_db_and_jsonl(self):
        row = record_ai_feedback_event(
            event_type="ai_gate_decision",
            level="info",
            config=self.cfg,
            account_alias="rortigoza",
            account_service="bingx",
            symbol="BTCUSDT",
            strategy="alloc_long",
            allow=True,
            risk_mult=0.8,
            reason="good setup",
            payload={"sig_score": 0.91},
            path_override=self.rel_stream,
        )
        self.assertIsNotNone(row.id)
        self.assertEqual(AiFeedbackEvent.objects.count(), 1)
        stream_path = Path(settings.BASE_DIR) / self.rel_stream
        self.assertTrue(stream_path.exists())
        lines = stream_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        parsed = json.loads(lines[0])
        self.assertEqual(parsed.get("ev"), "ai_gate_decision")
        self.assertEqual(parsed.get("sym"), "BTCUSDT")

    @override_settings(AI_FEEDBACK_JSONL_ENABLED=True, AI_FEEDBACK_JSONL_MAX_BYTES=2_000_000)
    def test_load_feedback_context_tail_respects_budget(self):
        for idx in range(5):
            record_ai_feedback_event(
                event_type="order_send_error",
                level="error",
                config=self.cfg,
                account_alias="rortigoza",
                account_service="bingx",
                symbol="SOLUSDT",
                strategy="alloc_long",
                allow=False,
                risk_mult=0.0,
                reason=f"boom-{idx}",
                payload={"idx": idx},
                path_override=self.rel_stream,
            )
        text, used, estimated = load_feedback_context_tail(
            model_name="gpt-5",
            max_tokens=120,
            path_override=self.rel_stream,
        )
        self.assertGreater(used, 0)
        self.assertTrue(bool(text.strip()))
        self.assertIn("boom-4", text)
        self.assertIsInstance(estimated, bool)

    @override_settings(AI_FEEDBACK_JSONL_ENABLED=True, AI_FEEDBACK_JSONL_MAX_BYTES=2_000_000)
    def test_rebuild_feedback_stream_command(self):
        for idx in range(3):
            record_ai_feedback_event(
                event_type="ai_gate_error",
                level="error",
                config=self.cfg,
                account_alias="eudy",
                account_service="bingx",
                symbol="ETHUSDT",
                strategy="alloc_short",
                allow=True,
                risk_mult=1.0,
                reason=f"err-{idx}",
                payload={"idx": idx},
                path_override=self.rel_stream,
            )
        out_rel = (self.tmp_dir / "rebuilt.jsonl").relative_to(settings.BASE_DIR).as_posix()
        call_command(
            "rebuild_ai_feedback_stream",
            "--hours",
            "240",
            "--limit",
            "100",
            "--write",
            out_rel,
        )
        out_path = Path(settings.BASE_DIR) / out_rel
        self.assertTrue(out_path.exists())
        lines = out_path.read_text(encoding="utf-8").splitlines()
        self.assertGreaterEqual(len(lines), 2)
        meta = json.loads(lines[0])
        self.assertIn("_meta", meta)
