FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data ./data
COPY scripts ./scripts
COPY models ./models

EXPOSE 8000

# Shell form so $PORT actually expands -- Render (and similar platforms) assign
# their own PORT env var and scan for the app listening on THAT port, not
# whatever's hardcoded here. Falls back to 8000 for a plain local `docker run`.
CMD sh -c "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"
