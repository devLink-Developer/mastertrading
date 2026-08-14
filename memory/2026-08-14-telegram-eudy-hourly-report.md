# DEBUG REPORT — Telegram Eudy hourly report

- **Symptom:** Eudy performance summaries were sent every 180 minutes and exposed technical counters before the account result, making them hard to read.
- **Root cause:** Each stack had an independent `control_performance_report` row, but both rows retained the 180-minute defaults. The report template in `risk/tasks.py` prioritized raw module and allocator counter names over equity, realized/unrealized PnL, outcomes, and positions.
- **Separation evidence:** Ricardo and Eudy use the same Telegram bot token but different `TELEGRAM_CHAT_ID` values and separate PostgreSQL runtime-control rows.
- **Fix:** Eudy runtime control changed to interval mode with `beat_minutes=60` and `window_minutes=60`. The report template now leads with account alias, equity change, PnL, outcomes, open positions, and plain-language activity labels.
- **Regression test:** `risk/tests_performance_report_message.py`.
- **Verification:** Targeted report/control tests passed (3/3); Ruff and compile checks passed. Deployed Eudy worker revision `92cb777`. A real Telegram test message returned `SEND_OK=True`.
- **Isolation:** Ricardo runtime control remained at 180 minutes and its containers were not restarted.
- **Status:** DONE
