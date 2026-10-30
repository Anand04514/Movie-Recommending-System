# syntax=docker/dockerfile:1

# --- 1. Build the React app ---------------------------------------------------------------
# The bundle is served by the API's own origin (SERVE_FRONTEND=1), so production needs one
# container, one port and no CORS configuration.
FROM node:22-alpine AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# --- 2. Runtime ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# uv resolves the exact versions in uv.lock; pinning the binary itself would be nice, but the
# lockfile is what makes the Python dependencies reproducible.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    PROCESSED_DIR=/app/data/processed \
    SERVE_FRONTEND=1

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY alembic.ini ./
COPY migrations ./migrations
COPY backend ./backend
COPY recsys ./recsys
COPY scripts ./scripts
COPY --from=frontend /app/frontend/dist ./frontend/dist

# The catalog and model artifacts are published as a checksummed bundle and downloaded on start
# (scripts/fetch_artifacts.py), because they are derived data and are not in the repository.
RUN mkdir -p /app/data/processed

# Run unprivileged: nothing here needs to write outside /app/data/processed.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data/processed \
    && chown -R appuser:appuser /app/data
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"

# bash is in the slim image and avoids relying on the executable bit, which a Windows checkout
# does not necessarily preserve.
CMD ["bash", "scripts/start.sh"]
