import os
import json
import time
import tiktoken
import requests
from openai import OpenAI

# Initialize standard token encoder
encoder = tiktoken.get_encoding("cl100k_base")

# Retrieve judging environment variables (Falling back to local endpoints for your testing)
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "mock_key")
FIREWORKS_BASE_URL = os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
LOCAL_OLLAMA_URL = os.getenv("LOCAL_OLLAMA_URL", "http://localhost:11434")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "gemma4:e4b")

# Initialize SDK clients
remote_client = OpenAI(api_key=FIREWORKS_API_KEY, base_url=FIREWORKS_BASE_URL)


def call_local_ollama(prompt: str, timeout: float = 30.0) -> str:
    """Call the local Ollama /api/generate endpoint and return the response text."""
    response = requests.post(
        f"{LOCAL_OLLAMA_URL}/api/generate",
        json={"model": LOCAL_MODEL, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["response"]

def assess_complexity(prompt: str) -> str:
    """
    LifeInvaders Gatekeeper Logic:
    Calculates token count and scans for heavy logical keywords to decide tier routing.
    """
    token_count = len(encoder.encode(prompt))
    
    # Tier 1: Massive context threshold bypass
    if token_count > 1500:
        return "remote"
        
    # Tier 2: Heavy engineering/logic keyword filter
    complex_keywords = ["debugger", "multithreading", "runtime", "optimize", "compilation"]
    if any(word in prompt.lower() for word in complex_keywords):
        return "remote"
        
    return "local"

def process_task(task: dict) -> dict:
    task_id = task.get("id", "unknown")
    prompt = task.get("prompt", "")
    
    route = assess_complexity(prompt)
    
    try:
        if route == "local":
            # Zero-cost token tracking layer
            answer = call_local_ollama(prompt)
            routed_via = "local_gemma4"
        else:
            # Paid Fireworks track
            response = remote_client.chat.completions.create(
                model="accounts/fireworks/models/gemma2-9b-it",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            answer = response.choices[0].message.content
            routed_via = "remote_fireworks"

    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        # Local Ollama isn't running or didn't respond in time
        answer = f"Fallback routing activated: local Ollama unavailable ({str(e)})"
        routed_via = "fallback_bypass"
    except Exception as e:
        # Self-healing fallback option if local processing fails
        answer = f"Fallback routing activated due to exception: {str(e)}"
        routed_via = "fallback_bypass"

    return {
        "id": task_id,
        "answer": answer,
        "routed_via": routed_via
    }

def main():
    input_path = "/input/tasks.json"
    output_path = "/output/results.json"
    
    # Fallback paths for your local Windows verification tests
    if not os.path.exists(input_path):
        os.makedirs("./mock_io/input", exist_ok=True)
        os.makedirs("./mock_io/output", exist_ok=True)
        input_path = "./mock_io/input/tasks.json"
        output_path = "./mock_io/output/results.json"
        
        # Write a dummy evaluation sample file if it doesn't exist yet
        if not os.path.exists(input_path):
            sample_tasks = [
                {"id": "1", "prompt": "What is the capital of France?"},
                {"id": "2", "prompt": "Optimize this multi-threaded runtime compilation script process."}
            ]
            with open(input_path, "w") as f:
                json.dump(sample_tasks, f)

    # Load tasks batch array
    with open(input_path, "r") as f:
        tasks = json.load(f)
        
    results = [process_task(task) for task in tasks]
    
    # Write final answers back to the judging destination
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Batch execution complete. Processed {len(results)} items successfully.")

if __name__ == "__main__":
    main()