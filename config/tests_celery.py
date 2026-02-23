from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

from django.test import SimpleTestCase

from config import celery as celery_cfg


class CeleryFailureHandlingTest(SimpleTestCase):
    @patch.dict(
        os.environ,
        {"CELERY_DLQ_REDIS_KEY": "celery:test:dlq", "CELERY_DLQ_MAXLEN": "123"},
        clear=False,
    )
    def test_push_task_failure_dlq_writes_payload(self):
        pipe = Mock()
        client = Mock()
        client.pipeline.return_value = pipe
        with patch("config.celery._dlq_client", return_value=client):
            celery_cfg._push_task_failure_dlq({"task_name": "t1", "error": "boom"})

        pipe.lpush.assert_called_once()
        pipe.ltrim.assert_called_once_with("celery:test:dlq", 0, 122)
        pipe.execute.assert_called_once()

    def test_task_failure_signal_routes_to_dlq_and_notify(self):
        sender = SimpleNamespace(name="signals.tasks.run_signal_engine")
        with (
            patch("config.celery._push_task_failure_dlq") as push_mock,
            patch("config.celery._notify_task_failure") as notify_mock,
        ):
            celery_cfg._on_task_failure(
                sender=sender,
                task_id="abc-1",
                exception=RuntimeError("db-down"),
                args=[],
                kwargs={"k": "v"},
                einfo=SimpleNamespace(traceback="tb"),
            )

        push_mock.assert_called_once()
        notify_mock.assert_called_once_with(
            "signals.tasks.run_signal_engine",
            "abc-1",
            "db-down",
        )

    @patch.dict(
        os.environ,
        {"CELERY_NOTIFY_ON_FAILURE": "true", "CELERY_FAILURE_NOTIFY_THROTTLE_SECONDS": "300"},
        clear=False,
    )
    def test_notify_task_failure_is_throttled_by_redis_key(self):
        client = Mock()
        client.set.return_value = False
        with (
            patch("config.celery._dlq_client", return_value=client),
            patch("risk.notifications.notify_error") as notify_mock,
        ):
            celery_cfg._notify_task_failure("execution.tasks.execute_orders", "id-1", "error")

        notify_mock.assert_not_called()

    @patch.dict(
        os.environ,
        {"CELERY_NOTIFY_ON_FAILURE": "true", "CELERY_FAILURE_NOTIFY_THROTTLE_SECONDS": "300"},
        clear=False,
    )
    def test_notify_task_failure_sends_alert_when_not_throttled(self):
        client = Mock()
        client.set.return_value = True
        with (
            patch("config.celery._dlq_client", return_value=client),
            patch("risk.notifications.notify_error") as notify_mock,
        ):
            celery_cfg._notify_task_failure("execution.tasks.execute_orders", "id-2", "boom")

        notify_mock.assert_called_once_with("celery:execution.tasks.execute_orders", "id-2: boom")
