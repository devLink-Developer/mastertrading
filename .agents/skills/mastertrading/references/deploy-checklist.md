# Deploy Checklist

## Main stack

1. `git status`
2. `git add -A`
3. `git commit -m "..."`
4. `git push origin main`
5. In server `/opt/trading_bot`:
   - `git checkout main`
   - `git pull --ff-only origin main`
6. Deploy:
   - `sudo -n docker compose up -d --build web worker beat`
7. Verify:
   - `sudo -n docker compose ps`
   - `sudo -n docker compose logs --tail=120 worker`

## Eudy stack

1. In server `/opt/trading_bot`:
   - `git checkout main`
   - `git pull --ff-only origin main`
2. Deploy:
   - `sudo -n docker compose -p trading_bot_eudy -f docker-compose.eudy.yml --env-file .env.eudy up -d --build web worker beat`
3. Verify:
   - `sudo -n docker compose -p trading_bot_eudy -f docker-compose.eudy.yml --env-file .env.eudy ps`
   - `sudo -n docker compose -p trading_bot_eudy -f docker-compose.eudy.yml --env-file .env.eudy logs --tail=120 worker`

## Post-deploy sanity checks

- runtime settings from Django shell if config changed
- targeted tests for touched areas when practical
- smoke command for management commands if a new one was added
- compare stack parity when behavior should match
