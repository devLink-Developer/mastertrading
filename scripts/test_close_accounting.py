"""Run targeted Django tests with SQLite and fail on any network attempt."""
from contextlib import ExitStack
import os
from pathlib import Path
import socket
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["DJANGO_SETTINGS_MODULE"] = "execution.accounting_test_settings"
attempts = []


def block_network(*args, **kwargs):
    attempts.append("outbound network blocked")
    raise RuntimeError("Outbound network is disabled in isolated accounting tests")


if __name__ == "__main__":
    labels = sys.argv[1:] or [
        "execution.test_close_evidence",
        "execution.tests_close_accounting",
        "execution.tests_execution_accounting",
        "execution.tests_tasks",
        "execution.tests_accounting_status",
        "signals.tests_dynamic_weights",
        "signals.tests_meta_allocator",
        "risk.tests_report_controls",
        "risk.tests_pending_notifications",
    ]
    with ExitStack() as stack:
        stack.enter_context(patch.object(socket.socket, "connect", block_network))
        stack.enter_context(patch.object(socket.socket, "connect_ex", block_network))
        stack.enter_context(patch.object(socket, "create_connection", block_network))
        import django
        django.setup()
        from django.conf import settings
        from django.test.utils import get_runner
        assert settings.MODE == "demo" and settings.TRADING_ENABLED is False
        assert settings.DATABASES["default"]["ENGINE"] == "django.db.backends.sqlite3"
        assert settings.DATABASES["default"]["NAME"] == ":memory:"
        assert settings.CELERY_TASK_ALWAYS_EAGER is False
        runner = get_runner(settings)(verbosity=2, interactive=False)
        failures = runner.run_tests(labels)
        print(f"Network attempts: {len(attempts)}")
        raise SystemExit(1 if failures or attempts else 0)
