"""Isolated Django settings for accounting tests; no runtime services."""
import os
from unittest.mock import patch

os.environ.update({
    "MODE": "demo", "TRADING_ENABLED": "false", "USE_SQLITE": "true",
    "DEBUG": "false", "SECRET_KEY": "local-isolated-accounting-tests-only",
    "REDIS_URL": "memory://", "BINGX_SANDBOX": "true",
    "EXCHANGE_ACCOUNT_SANDBOX": "true", "CELERY_NOTIFY_ON_FAILURE": "false",
    "TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": "", "PYTHON_DOTENV_DISABLED": "1",
})

# Prevent importing any workspace .env, including older dotenv versions.
with patch("dotenv.load_dotenv", return_value=False):
    from config.settings import *  # noqa: F403,F401

MODE = "demo"
TRADING_ENABLED = False
DATABASES = {"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": ":memory:",
                          "TEST": {"NAME": ":memory:"}}}
CACHES = {"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache",
                       "LOCATION": "isolated-accounting-tests"}}
CELERY_BROKER_URL = "memory://"
CELERY_RESULT_BACKEND = "cache+memory://"
CELERY_TASK_ALWAYS_EAGER = False
CELERY_BEAT_SCHEDULE = {}
EMAIL_BACKEND = "django.core.mail.backends.locmem.EmailBackend"
AI_FEEDBACK_JSONL_PATH = str(BASE_DIR / "tmp" / "profit_fix_env" / "runtime" / "feedback_stream.jsonl")
ROOT_URLCONF = "execution.accounting_test_settings"
urlpatterns = []
LOGGING = {"version": 1, "disable_existing_loggers": False,
           "handlers": {"console": {"class": "logging.StreamHandler"}},
           "root": {"handlers": ["console"], "level": "WARNING"}}
