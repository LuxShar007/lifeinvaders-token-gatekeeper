FROM python:3.11-slim

WORKDIR /app

# Prevent Python from writing pyc files to disk and ensure real-time streaming logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire directory structure (including your prompts/ and mock_io/ folders)
COPY . .

# Mount points specified by the lablab.ai environment contract
VOLUME ["/input", "/output"]

# Inform the platform that the container listens on port 8000
EXPOSE 8000

# Run the live web server app using uvicorn instead of a single python execution
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]