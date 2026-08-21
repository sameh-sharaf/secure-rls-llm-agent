FROM python:3.11-slim

# Non-root by default. The app only ever needs to read its own database file.
RUN useradd --create-home --uid 10001 app

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the fixtures at image-build time so the container starts ready. The
# dataset is seeded, so this is reproducible: the same image always contains
# the same 1000 rows and the same canaries.
RUN python scripts/generate_data.py \
 && python scripts/build_db.py \
 && python scripts/build_index.py \
 && chown -R app:app /app

USER app

ENV SECURE_RLS_MODEL=llama3.1:8b \
    OLLAMA_HOST=http://ollama:11434 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["python", "-m", "streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
