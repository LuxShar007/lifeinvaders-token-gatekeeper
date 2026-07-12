FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Make the entrypoint script executable
RUN chmod +x entrypoint.sh

VOLUME ["/input", "/output"]
EXPOSE 8000

# Execute the entrypoint script wrapper
CMD ["./entrypoint.sh"]