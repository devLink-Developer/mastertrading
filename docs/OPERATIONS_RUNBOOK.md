# Operations Runbook

Last update: 2026-03-01

Scope
- Trading project (`mastertrading`) production operations.
- Main and Eudy stacks on the same server.

Server access
- Host: `200.58.107.187`
- SSH port: `5344`
- User: `rortigoza`

## 1) Local git flow (before deploy)
```powershell
cd C:\Users\rortigoza\Documents\Proyectos\mastertrading
git checkout main
git pull --ff-only origin main
git status
git add -A
git commit -m "your message"
git push origin main
```

## 2) SSH into server
Linux/macOS:
```bash
ssh -p 5344 rortigoza@200.58.107.187
```

Windows (PuTTY plink):
```powershell
& "C:\Program Files\PuTTY\plink.exe" -ssh -P 5344 -l rortigoza 200.58.107.187
```

## 3) Main stack deploy (`/opt/trading_bot`)
```bash
cd /opt/trading_bot
sudo -n git checkout main
sudo -n git pull --ff-only origin main
sudo -n git rev-parse --short HEAD

sudo -n docker compose up -d --build
sudo -n docker compose ps
```

## 4) Eudy stack deploy (same repo, different compose project)
```bash
cd /opt/trading_bot
sudo -n docker compose -p trading_bot_eudy \
  -f docker-compose.eudy.yml \
  -f docker-compose.eudy.override.yml \
  up -d --build

sudo -n docker compose -p trading_bot_eudy \
  -f docker-compose.eudy.yml \
  -f docker-compose.eudy.override.yml \
  ps
```

## 5) Health checks
Main:
```bash
cd /opt/trading_bot
sudo -n docker compose ps
sudo -n docker compose logs --tail=150 worker
sudo -n docker compose logs --tail=150 web
```

Eudy:
```bash
cd /opt/trading_bot
sudo -n docker compose -p trading_bot_eudy \
  -f docker-compose.eudy.yml \
  -f docker-compose.eudy.override.yml \
  logs --tail=150 worker
```

## 6) Quick runtime checks (Django shell)
Main:
```bash
cd /opt/trading_bot
sudo -n docker compose exec -T web python manage.py shell -c "
from execution.models import Position, OperationReport;
print('open_positions=', Position.objects.filter(is_open=True).count());
print('reports=', OperationReport.objects.count())"
```

Eudy:
```bash
cd /opt/trading_bot
sudo -n docker compose -p trading_bot_eudy -f docker-compose.eudy.yml -f docker-compose.eudy.override.yml exec -T web python manage.py shell -c "
from execution.models import Position, OperationReport;
print('open_positions=', Position.objects.filter(is_open=True).count());
print('reports=', OperationReport.objects.count())"
```

## 7) Rollback (fast)
```bash
cd /opt/trading_bot
sudo -n git log --oneline -n 5
sudo -n git checkout <previous_commit_sha>
sudo -n docker compose up -d --build
sudo -n docker compose -p trading_bot_eudy -f docker-compose.eudy.yml -f docker-compose.eudy.override.yml up -d --build
```

Then pin/restore branch state:
```bash
cd /opt/trading_bot
sudo -n git checkout main
```

## 8) Incident checklist
1. Confirm services are `Up`.
2. Check worker logs for exceptions in last 15-60 min.
3. Verify lock and queue progress (no task stall).
4. Inspect latest `OperationReport` close reasons.
5. If exchange/API unstable: set `TRADING_ENABLED=false` and redeploy.

## 9) Quant/LLM guardrails checks
Validate TOON files before enabling new AI context:
```bash
cd /opt/trading_bot
sudo -n docker compose exec -T web python manage.py validate_toon_context --glob "docs/*.toon.md" --strict
```

Run regime-aware Monte Carlo manually:
```bash
cd /opt/trading_bot
sudo -n docker compose exec -T web python manage.py monte_carlo --days 90 --sims 10000 --regime-aware --stress-profile balanced --json reports/monte_carlo/manual_latest.json
```

## 10) Security and secrets
- Keep API keys/passwords only in server/local `.env`.
- Never commit credentials or dumps to remote.
- Sanitize logs/reports before sharing externally.
