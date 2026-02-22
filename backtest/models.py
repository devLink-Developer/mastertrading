"""
Backtest models – stores backtest run metadata and per-trade results
so they can be queried from Django admin or the API.
"""
from django.db import models
from core.models import Instrument


class BacktestRun(models.Model):
    """One execution of the backtester."""

    class Status(models.TextChoices):
        RUNNING = "running", "Running"
        DONE = "done", "Done"
        ERROR = "error", "Error"

    name = models.CharField(max_length=128, blank=True, default="")
    status = models.CharField(max_length=10, choices=Status.choices, default=Status.RUNNING)
    start_date = models.DateTimeField()
    end_date = models.DateTimeField()
    instruments_json = models.JSONField(default=list, help_text="List of symbols tested")
    settings_json = models.JSONField(default=dict, help_text="Snapshot of settings used")
    metrics_json = models.JSONField(default=dict, help_text="Summary metrics after run")
    created_at = models.DateTimeField(auto_now_add=True)
    duration_seconds = models.FloatField(default=0.0)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Backtest {self.id} ({self.start_date:%Y-%m-%d} → {self.end_date:%Y-%m-%d}) {self.status}"


class BacktestTrade(models.Model):
    """One simulated trade inside a backtest run."""

    class Outcome(models.TextChoices):
        WIN = "win", "Win"
        LOSS = "loss", "Loss"
        BE = "be", "Break-even"

    run = models.ForeignKey(BacktestRun, on_delete=models.CASCADE, related_name="trades")
    instrument = models.ForeignKey(Instrument, on_delete=models.CASCADE)
    side = models.CharField(max_length=5)  # buy / sell
    qty = models.FloatField()
    entry_price = models.FloatField()
    exit_price = models.FloatField()
    entry_ts = models.DateTimeField()
    exit_ts = models.DateTimeField()
    pnl_abs = models.FloatField()
    pnl_pct = models.FloatField()
    fee_paid = models.FloatField(default=0.0)
    reason = models.CharField(max_length=32)  # tp / sl / trailing / signal_flip / end_of_data
    score = models.FloatField(default=0.0)
    outcome = models.CharField(max_length=4, choices=Outcome.choices, default=Outcome.BE)

    class Meta:
        ordering = ["entry_ts"]

    def __str__(self):
        return f"{self.instrument.symbol} {self.side} {self.outcome} {self.pnl_pct:+.4f}"
