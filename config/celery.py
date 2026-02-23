from __future__ import annotations

import json
import logging
import os

from celery import Celery
from celery.signals import task_failure
import redis

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

app = Celery("mastertrading")
app.config_from_object("django.conf:settings", namespace="CELERY")
logger = logging.getLogger(__name__)


app.conf.update(
    task_acks_late=os.getenv("CELERY_TASK_ACKS_LATE", "true").lower() == "true",
    task_reject_on_worker_lost=os.getenv("CELERY_TASK_REJECT_ON_WORKER_LOST", "true").lower() == "true",
    task_time_limit=int(os.getenv("CELERY_TASK_TIME_LIMIT", "300")),
    task_soft_time_limit=int(os.getenv("CELERY_TASK_SOFT_TIME_LIMIT", "240")),
    task_default_queue=os.getenv("CELERY_TASK_DEFAULT_QUEUE", "celery"),
)
app.autodiscover_tasks()


@app.task(bind=True)
def debug_task(self):
    print(f"Debug task executed {self.request!r}")


def _dlq_client() -> redis.Redis | None:
    try:
        broker_url = str(app.conf.broker_url or "")
        if not broker_url.startswith("redis://"):
            return None
        return redis.from_url(broker_url)
    except Exception:
        return None


def _push_task_failure_dlq(payload: dict) -> None:
    client = _dlq_client()
    if client is None:
        return
    key = os.getenv("CELERY_DLQ_REDIS_KEY", "celery:dlq")
    maxlen = max(100, int(os.getenv("CELERY_DLQ_MAXLEN", "2000") or "2000"))
    try:
        encoded = json.dumps(payload, default=str)
        pipe = client.pipeline()
        pipe.lpush(key, encoded)
        pipe.ltrim(key, 0, maxlen - 1)
        pipe.execute()
    except Exception as exc:
        logger.warning("Failed to push task failure to DLQ: %s", exc)


def _notify_task_failure(task_name: str, task_id: str, message: str) -> None:
    if os.getenv("CELERY_NOTIFY_ON_FAILURE", "true").lower() != "true":
        return
    client = _dlq_client()
    throttle_seconds = max(
        30,
        int(os.getenv("CELERY_FAILURE_NOTIFY_THROTTLE_SECONDS", "300") or "300"),
    )
    throttle_key = f"celery:notify_fail:{task_name}"
    if client is not None:
        try:
            if not client.set(throttle_key, "1", nx=True, ex=throttle_seconds):
                return
        except Exception:
            pass
    try:
        from risk.notifications import notify_error

        notify_error(f"celery:{task_name}", f"{task_id}: {message}")
    except Exception as exc:
        logger.warning("Celery failure notification failed: %s", exc)


@task_failure.connect
def _on_task_failure(
    sender=None,
    task_id=None,
    exception=None,
    args=None,
    kwargs=None,
    traceback=None,
    einfo=None,
    **extras,
):
    task_name = getattr(sender, "name", str(sender or "unknown"))
    error_text = str(exception or "unknown error")
    payload = {
        "task_name": task_name,
        "task_id": str(task_id or ""),
        "error": error_text,
        "args": args or [],
        "kwargs": kwargs or {},
        "traceback": str(getattr(einfo, "traceback", "") or "")[:4000],
    }
    _push_task_failure_dlq(payload)
    _notify_task_failure(task_name, str(task_id or ""), error_text)
