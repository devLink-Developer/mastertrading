from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from core.api_runtime import build_optimized_context, get_active_api_config


class Command(BaseCommand):
    help = "Preview optimized prompt context from admin-configured context files."

    def add_arguments(self, parser):
        parser.add_argument("--alias", type=str, default="", help="ApiProviderConfig.name_alias")
        parser.add_argument("--provider", type=str, default="", help="Provider filter (openai, openrouter, ...)")
        parser.add_argument("--owner-id", type=int, default=0, help="Optional owner user id")
        parser.add_argument("--prompt-file", type=str, default="", help="Path to user prompt file")
        parser.add_argument("--prompt-text", type=str, default="", help="Raw user prompt text")
        parser.add_argument("--reserve-output", type=int, default=-1, help="Reserved output token budget")
        parser.add_argument("--write", type=str, default="", help="Optional output file path")

    def handle(self, *args, **options):
        alias = str(options["alias"] or "").strip()
        provider = str(options["provider"] or "").strip().lower()
        owner_id = int(options["owner_id"] or 0) or None
        prompt_file = str(options["prompt_file"] or "").strip()
        prompt_text = str(options["prompt_text"] or "")
        reserve = int(options["reserve_output"])
        out_file = str(options["write"] or "").strip()

        if prompt_file:
            try:
                prompt_text = Path(prompt_file).read_text(encoding="utf-8")
            except Exception as exc:
                raise CommandError(f"cannot read --prompt-file: {exc}") from exc

        cfg = get_active_api_config(
            provider=provider or None,
            owner_id=owner_id,
            name_alias=alias or None,
        )
        if not cfg:
            raise CommandError("No active ApiProviderConfig found for the filters.")

        result = build_optimized_context(
            cfg,
            user_prompt=prompt_text,
            reserve_output_tokens=(None if reserve < 0 else reserve),
        )

        self.stdout.write("=" * 72)
        self.stdout.write(f"Config      : {cfg.name_alias} ({cfg.provider}/{cfg.model_name})")
        self.stdout.write(f"Owner       : {getattr(cfg.owner, 'username', 'global')}")
        self.stdout.write(f"User tokens : {result.user_tokens}")
        self.stdout.write(f"Ctx tokens  : {result.used_context_tokens}")
        self.stdout.write(f"Prompt total: {result.total_prompt_tokens}")
        self.stdout.write(f"Input budget: {result.available_input_tokens}")
        self.stdout.write(f"Reserved out: {result.reserve_output_tokens}")
        self.stdout.write(f"Estimated   : {result.estimated}")
        self.stdout.write("-" * 72)
        for row in result.files:
            status = "SKIP" if row.skipped else "USE "
            self.stdout.write(
                f"[{status}] id={row.context_file_id} tok={row.used_tokens}/{row.allocated_tokens} "
                f"path={row.file_path} reason={row.reason or '-'}"
            )
        self.stdout.write("=" * 72)

        if out_file:
            Path(out_file).write_text(result.context_text, encoding="utf-8")
            self.stdout.write(self.style.SUCCESS(f"Optimized context written to: {out_file}"))
        else:
            self.stdout.write(result.context_text)
