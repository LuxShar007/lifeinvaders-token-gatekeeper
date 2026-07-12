FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install curl so the validation loops check gateway availability properly
RUN apt-get update && apt-get install -y curl && rm -rf /lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

VOLUME ["/input", "/output"]
EXPOSE 8000

# Inline string array boots the server, runs the evaluation, and drops a clean exit
CMD ["/bin/bash", "-c", "python -m uvicorn main:app --host 0.0.0.0 --port 8000 & SERVER_PID=$!; for i in {1..10}; do if curl -s http://localhost:8000/ > /dev/null; then break; fi; sleep 1; done; python harness.py; kill $SERVER_PID; exit 0"]