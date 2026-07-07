FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Mount points specified by the lablab.ai environment contract
VOLUME ["/input", "/output"]

CMD ["python", "main.py"]