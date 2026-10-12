# ---------------------------------------------------------------------------
# Asteroids-AI – Streamlit container
# ---------------------------------------------------------------------------
FROM python:3.11-slim

# System deps (none required beyond pip, but keep layer clean)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all game sources
COPY . .

# Streamlit must listen on 0.0.0.0 so Docker can expose it
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501
# Disable the "check for updates" network call at boot
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
# Suppress the email prompt on first run
ENV STREAMLIT_SERVER_HEADLESS=true

EXPOSE 8501

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
