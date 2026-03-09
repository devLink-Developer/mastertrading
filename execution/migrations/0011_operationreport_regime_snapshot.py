from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("execution", "0010_operationreport_mfe_metrics"),
    ]

    operations = [
        migrations.AddField(
            model_name="operationreport",
            name="monthly_regime",
            field=models.CharField(blank=True, default="", max_length=24),
        ),
        migrations.AddField(
            model_name="operationreport",
            name="weekly_regime",
            field=models.CharField(blank=True, default="", max_length=24),
        ),
        migrations.AddField(
            model_name="operationreport",
            name="daily_regime",
            field=models.CharField(blank=True, default="", max_length=24),
        ),
        migrations.AddField(
            model_name="operationreport",
            name="btc_lead_state",
            field=models.CharField(blank=True, default="", max_length=24),
        ),
        migrations.AddField(
            model_name="operationreport",
            name="recommended_bias",
            field=models.CharField(blank=True, default="", max_length=24),
        ),
    ]
