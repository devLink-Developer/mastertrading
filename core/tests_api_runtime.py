from __future__ import annotations

import shutil
from pathlib import Path

from django.conf import settings
from django.test import TestCase

from core.api_runtime import (
    _resolve_safe_context_path,
    build_optimized_context,
    get_active_api_config,
    log_token_usage,
)
from core.models import ApiContextFile, ApiProviderConfig, ApiTokenUsageLog


class ApiRuntimeTest(TestCase):
    def setUp(self):
        self.cfg = ApiProviderConfig.objects.create(
            name_alias="gpt-main",
            provider=ApiProviderConfig.Provider.OPENAI,
            model_name="gpt-5",
            active=True,
            is_default=True,
            max_input_tokens=200,
            max_output_tokens=40,
            api_key="dummy",
        )
        self.tmp_dir = Path(settings.BASE_DIR) / "tmp" / "tests_api_runtime"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_get_active_api_config_returns_default(self):
        ApiProviderConfig.objects.create(
            name_alias="gpt-secondary",
            provider=ApiProviderConfig.Provider.OPENAI,
            model_name="gpt-5-mini",
            active=True,
            is_default=False,
            api_key="dummy2",
        )
        picked = get_active_api_config(provider="openai")
        self.assertIsNotNone(picked)
        assert picked is not None
        self.assertEqual(picked.name_alias, "gpt-main")

    def test_get_active_api_config_owner_fallbacks_to_global(self):
        picked = get_active_api_config(provider="openai", owner_id=999999)
        self.assertIsNotNone(picked)
        assert picked is not None
        self.assertEqual(picked.name_alias, "gpt-main")

    def test_build_optimized_context_with_file_budget(self):
        file_a = self.tmp_dir / "a.md"
        file_b = self.tmp_dir / "b.md"
        file_a.write_text(("A " * 800).strip(), encoding="utf-8")
        file_b.write_text(("B " * 800).strip(), encoding="utf-8")

        rel_a = file_a.relative_to(settings.BASE_DIR).as_posix()
        rel_b = file_b.relative_to(settings.BASE_DIR).as_posix()
        ApiContextFile.objects.create(
            config=self.cfg,
            name="Core Rules",
            file_path=rel_a,
            required=True,
            priority=1,
            max_tokens=60,
            trim_mode=ApiContextFile.TrimMode.HEAD,
        )
        ApiContextFile.objects.create(
            config=self.cfg,
            name="Ops",
            file_path=rel_b,
            required=False,
            priority=2,
            max_tokens=120,
            trim_mode=ApiContextFile.TrimMode.TAIL,
        )
        result = build_optimized_context(self.cfg, user_prompt="hola mundo", reserve_output_tokens=50)
        self.assertGreaterEqual(result.available_input_tokens, 1)
        self.assertLessEqual(result.used_context_tokens, result.available_input_tokens)
        self.assertIn("Core Rules", result.context_text)
        self.assertTrue(any(not row.skipped for row in result.files))

    def test_resolve_safe_context_path_blocks_outside_basedir(self):
        with self.assertRaises(ValueError):
            _resolve_safe_context_path("../outside.txt")

    def test_log_token_usage_persists_totals(self):
        row = log_token_usage(
            config=self.cfg,
            provider="openai",
            model_name="gpt-5",
            operation="unit_test",
            prompt_tokens=120,
            completion_tokens=30,
            context_tokens=70,
            estimated=True,
            metadata={"k": "v"},
        )
        self.assertIsNotNone(row.id)
        stored = ApiTokenUsageLog.objects.get(id=row.id)
        self.assertEqual(stored.total_tokens, 150)
        self.assertEqual(stored.context_tokens, 70)
        self.assertTrue(stored.estimated)
