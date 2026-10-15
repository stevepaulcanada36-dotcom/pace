FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
RUN pip install --no-cache-dir ".[dashboard,serve]"
RUN useradd -m app && mkdir -p /data && chown app /data
USER app
ENV PACESENSE_DB=/data/pacesense.sqlite3 \
    PACESENSE_DATA=/app/data/races.csv
EXPOSE 8501
# Dashboard by default. For the JSON API instead:
#   docker run -p 8000:8000 IMAGE gunicorn "pacesense.web:app_from_env()" -w 1 --threads 8 -b 0.0.0.0:8000
CMD ["streamlit", "run", "src/pacesense/dashboard/app.py", "--server.port", "8501", "--server.address", "0.0.0.0", \
     "--server.headless", "true", "--browser.gatherUsageStats", "false"]
