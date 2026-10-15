FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
COPY config ./config
RUN pip install --no-cache-dir ".[dashboard,serve]"
RUN useradd -m app && mkdir -p /data && chown app /data
USER app
ENV PACESENSE_DB=/data/pacesense.sqlite3 \
    PACESENSE_DATA=/app/data/races.csv \
    PACESENSE_CONFIG_DIR=/app/config
EXPOSE 8501
# Dashboard by default. PaceSense Live:  docker run -p 8000:8000 IMAGE pacesense live --host 0.0.0.0 --no-browser
# Reproduce every number:               docker run -v "$PWD/results:/app/results" IMAGE pacesense reproduce --out /app/results
# For the JSON API instead:
#   docker run -p 8000:8000 IMAGE gunicorn "pacesense.web:app_from_env()" -w 1 --threads 8 -b 0.0.0.0:8000
CMD ["streamlit", "run", "src/pacesense/dashboard/app.py", "--server.port", "8501", "--server.address", "0.0.0.0", \
     "--server.headless", "true", "--browser.gatherUsageStats", "false"]
