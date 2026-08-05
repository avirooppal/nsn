import os
import neurosleepnet as nsn
from openai import OpenAI

def main():
    api_key = os.environ.get("OLLAMA_API_KEY", "")
    ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "https://ollama.com/v1")
    model_name = os.environ.get("OLLAMA_MODEL", "nemotron-3-nano:30b")

    if not api_key:
        raise ValueError("Set OLLAMA_API_KEY environment variable before running.")

    # Clean up previous memory (optional)
    if os.path.exists("real_agent_memory.db"):
        os.remove("real_agent_memory.db")

    print(f"Initializing Ollama Cloud API Client ({model_name})...")

    # Initialize the OpenAI-compatible client pointed at Ollama Cloud
    base_client = OpenAI(
        base_url=ollama_base_url,
        api_key=api_key,
    )

    # Wrap the client with NeuroSleepNet to give it persistent memory
    print("Wrapping client with NeuroSleepNet...")
    client = nsn.init(base_client, namespace="ollama_agent", db_path="real_agent_memory.db")

    print("\n" + "="*50)
    print("NeuroSleepNet + Ollama Cloud Demo")
    print("="*50 + "\n")

    # 1. Give the agent some facts (NSN stores everything it observes)
    question1 = "Hi! Just so you know, Alice is the server administrator and the secret password is 'Neuro2026'."
    print(f"User: {question1}")
    print("Agent thinking...")

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": question1}]
    )
    answer1 = response.choices[0].message.content
    print(f"Agent: {answer1}\n")

    # 2. Ask a follow-up — NSN will inject the relevant memory automatically
    question2 = "Can you remind me what the password is?"
    print(f"User: {question2}")
    print("Agent thinking (and automatically recalling memory!)...")

    response2 = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": question2}]
    )
    answer2 = response2.choices[0].message.content
    print(f"Agent: {answer2}\n")

    # 3. Directly inspect what NSN stored
    print("="*50)
    print("Direct Memory Recall Inspection:")
    hits = client.recall("password", limit=4)
    for i, hit in enumerate(hits, 1):
        print(f"  {i}. [{hit['memory_type']}] {hit['content']}")

if __name__ == "__main__":
    main()
