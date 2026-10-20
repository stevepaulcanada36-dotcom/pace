# PaceSense Studio in a container.
#   docker build -t pacesense .
#   docker run -p 8000:8000 -v pacesense-data:/data pacesense            -> http://localhost:8000
# Your own race data:  docker run -p 8000:8000 -v "$PWD/data:/app/data:ro" -e PACESENSE_DATA=/app/data/races.csv pacesense
# Other entry points (same image):
#   docker run -p 8501:8501 pacesense streamlit run src/pacesense/dashboard/app.py --server.port 8501 --server.address 0.0.0.0
#   docker run -v "$PWD/results:/app/results" pacesense pacesense reproduce --out /app/results

FROM python:3.12-slim AS build
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip wheel --no-cache-dir --wheel-dir /wheels ".[studio,dashboard]"

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY --from=build /wheels /wheels
RUN pip install --no-index --find-links=/wheels "pacesense[studio,dashboard]" && rm -rf /wheels
COPY src ./src
COPY data ./data
COPY config ./config
RUN useradd --create-home --uid 10001 app && mkdir -p /data && chown app /data
USER app
ENV PACESENSE_DB=/data/pacesense.sqlite3 \
    PACESENSE_CONFIG_DIR=/app/config \
    PACESENSE_HOST=0.0.0.0 \
    PACESENSE_PORT=8000 \
    PACESENSE_LOG_FORMAT=json
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=2).status == 200 else 1)"
# One worker: sessions live in memory. Requests run in a thread pool; each session has its own lock.
CMD ["uvicorn", "--factory", "pacesense.studio.app:create_app_from_env", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--proxy-headers", "--no-access-log", "--timeout-graceful-shutdown", "10"]
