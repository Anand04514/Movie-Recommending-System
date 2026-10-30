#!/usr/bin/env bash
# Production entrypoint: artifacts, schema, optional seed, then the API.
#
# Everything is idempotent, because a hosted container restarts and redeploys regularly:
#   * artifacts are fetched only when missing (and always verified when fetched);
#   * migrations run to head;
#   * the catalog is seeded once, not on every boot.
#
# The interpreter is resolved explicitly so the script behaves the same inside the image (where
# .venv/bin is on PATH) and in a local checkout (where it usually is not).
set -euo pipefail

if [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
elif [ -x ".venv/Scripts/python.exe" ]; then
  PYTHON=".venv/Scripts/python.exe"
else
  PYTHON="${PYTHON:-python}"
fi

processed_dir="${PROCESSED_DIR:-data/processed}"

if [ -n "${ARTIFACTS_URL:-}" ]; then
  fetch_args=(--url "$ARTIFACTS_URL" --processed-dir "$processed_dir")
  if [ -n "${ARTIFACTS_SHA256:-}" ]; then
    fetch_args+=(--sha256 "$ARTIFACTS_SHA256")
  else
    echo "warning: ARTIFACTS_SHA256 is not set; the download will not be verified" >&2
  fi
  "$PYTHON" scripts/fetch_artifacts.py "${fetch_args[@]}"
else
  echo "ARTIFACTS_URL is not set; expecting artifacts already in $processed_dir"
fi

echo "Applying database migrations"
"$PYTHON" -m alembic upgrade head

if [ "${SEED_ON_START:-0}" = "1" ]; then
  echo "Seeding the catalog (skipped when it is already populated)"
  "$PYTHON" scripts/seed_db.py \
    --processed-dir "$processed_dir" \
    --demo-users "${DEMO_USERS:-0}" \
    --skip-if-seeded
fi

echo "Starting the API"
exec "$PYTHON" -m uvicorn backend.app.main:app \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8000}" \
  --proxy-headers \
  --forwarded-allow-ips='*'
