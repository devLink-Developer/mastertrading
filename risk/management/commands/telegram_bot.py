"""Management command to run the interactive Telegram bot."""
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Run the interactive Telegram bot (polling mode)"

    def handle(self, *args, **options):
        from risk.telegram_bot import run_bot

        self.stdout.write(self.style.SUCCESS("Starting Telegram interactive bot..."))
        run_bot()
