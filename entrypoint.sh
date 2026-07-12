#!/bin/bash
set -e

# 1. Start the FastAPI gateway server in the background
echo "⚡ Starting FastAPI Proxy Gateway Server..."
python -m uvicorn main:app --host 0.0.0.0 --port 8000 &
SERVER_PID=$!

# 2. Wait for the server to spin up completely
echo "⏳ Waiting for gateway availability..."
for i in {1..10}; do
    if curl -s http://localhost:8000/ > /dev/null; then
        echo "✅ Gateway server online."
        break
    fi
    sleep 1
done

# 3. Execute the evaluation benchmark harness to process all tasks
echo "📊 Running evaluation harness..."
python benchmark/harness.py

# 4. Clean up the background server process and exit cleanly
echo "🏁 Evaluation complete. Exiting container."
kill $SERVER_PID
exit 0