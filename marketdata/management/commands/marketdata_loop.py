import logging
import time

from django.core.management.base import BaseCommand

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Stub market data loop for WS/REST ingestion"

    def handle(self, *args, **options):
        self.stdout.write("Market data loop started (stub).")
        try:
            while True:
                logger.info("marketdata_loop heartbeat")
                time.sleep(5)
        except KeyboardInterrupt:
            self.stdout.write("Market data loop stopped.")
