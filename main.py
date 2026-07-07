import os
import json
import time
import tiktoken
from openai import OpenAI

# Initialize token encoder
encoder = tiktoken.get_encoding("cl100k_base")

# Grab environment endpoints
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "mock_key")
FIREWORKS_BASE_URL = os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
LOCAL_OLLAMA_URL = os.getenv("LOCAL_OLLAMA_URL", "http://localhost:11434/v1")

remote_client = OpenAI(api_key=FIREWORKS_API_KEY, base_url=FIREWORKS_BASE_URL)
local_client = OpenAI(api_key="ollama", base_url=LOCAL_OLLAMA_URL)

def calculate_complexity_score(prompt: str) -> float:
    """
    LifeInvaders Speculative Gatekeeper Matrix:
    Evaluates multi-signal criteria to return a complexity weight from 0.0 to 1.0.
    """
    score = 0.0
    token_count = len(encoder.encode(prompt))
    
    # Signal 1: Context Weight (Max penalty if > 1500 tokens)
    if token_count > 1500: return 1.0
    score += (token_count / 1500.0) * 0.3  # Up to 0.3 weight
    
    # Signal 2: Syntactic Code/Engineering Keywords
    code_triggers = ["def ", "function", "runtime", "multithreading", "memory leak", "pointer", "compile"]
    if any(trigger in prompt.lower() for trigger in code_triggers):
        score += 0.4
        
    # Signal 3: Advanced Cognitive Reasoning Hooks
    reasoning_triggers = ["step-by-step", "analyze the root cause", "systemic", "mathematical proof"]
    if any(trigger in prompt.lower() for trigger in reasoning_triggers):
        score += 0.3
        
    return min(score, 1.0)

def process_task(task: dict) -> dict:
    task_id = task.get("id", "unknown")
    prompt = task.get("prompt", "")
    
    complexity = calculate_complexity_score(prompt)
    
    # Leaderboard Threshold Strategy
    if complexity > 0.70:
        # High complexity -> Fast-track directly to premium cloud to protect accuracy
        return call_remote_track(task_id, prompt, "high_complexity_bypass")
        
    # Low-to-Medium Complexity -> Run the Speculative Local Cascade
    try:
        # Step 1: Query Local Gemma 4 (Costs 0 points on leaderboard)
        response = local_client.chat.completions.create(
            model="gemma4",
            messages=[
                {"role": "system", "content": "Return a strict JSON object containing an 'answer' string and a numeric 'confidence_score' between 1 and 5."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}, # Keeps output clean
            temperature=0.1
        )
        
        # Step 2: Parse and assess model's self-evaluation confidence
        result_data = json.loads(response.choices[0].message.content)
        confidence = int(result_data.get("confidence_score", 0))
        
        if confidence >= 4:
            return {
                "id": task_id,
                "answer": result_data.get("answer", ""),
                "routed_via": "local_gemma4_confident"
            }
        else:
            # Step 3: Low confidence fallback -> Cascade to Fireworks to protect accuracy
            return call_remote_track(task_id, prompt, f"local_uncertainty_cascade (Conf: {confidence})")
            
    except Exception as e:
        # Self-healing layer: If JSON formats break or local model fails, fallback to cloud immediately
        return call_remote_track(task_id, prompt, f"error_fallback_cascade ({type(e).__name__})")

def call_remote_track(task_id: str, prompt: str, reason: str) -> dict:
    try:
        response = remote_client.chat.completions.create(
            model="accounts/fireworks/models/gemma2-9b-it",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return {
            "id": task_id,
            "answer": response.choices[0].message.content,
            "routed_via": f"remote_fireworks_via_{reason}"
        }
    except Exception as e:
        return {"id": task_id, "answer": f"Fatal System Exception: {str(e)}", "routed_via": "fatal_bypass"}

def main():
    input_path = "/input/tasks.json"
    output_path = "/output/results.json"
    
    if not os.path.exists(input_path):
        os.makedirs("./mock_io/input", exist_ok=True)
        os.makedirs("./mock_io/output", exist_ok=True)
        input_path = "./mock_io/input/tasks.json"
        output_path = "./mock_io/output/results.json"
        if not os.path.exists(input_path):
            with open(input_path, "w") as f:
                json.dump([{"id": "1", "prompt": "What is the capital of France?"}], f)

    with open(input_path, "r") as f:
        tasks = json.load(f)
        
    results = [process_task(task) for task in tasks]
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Batch execution complete. Processed {len(results)} items successfully.")

if __name__ == "__main__":
    main()