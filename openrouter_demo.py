import os
import neurosleepnet as nsn
from openai import OpenAI

def main():
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY environment variable before running.")

    model_name = os.environ.get("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")

    # Clean up previous memory (optional)
    if os.path.exists("openrouter_memory.db"):
        os.remove("openrouter_memory.db")

    print("Initializing OpenRouter API Client...")

    # Initialize the OpenAI client with OpenRouter's base URL and your key
    base_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # Wrap the client with NeuroSleepNet using the init pattern
    print("Wrapping client with NeuroSleepNet...")
    client = nsn.init(base_client, namespace="nemotron_agent", db_path="openrouter_memory.db")

    print("\n" + "="*50)
    print("NeuroSleepNet + OpenRouter Demo")
    print("="*50 + "\n")

    # 1. Tell the agent some facts through normal conversation
    # NSN automatically observes and stores this interaction!
    question1 = "Hi! Just so you know, Alice is the server administrator and the secret password is 'Neuro2026'."
    print(f"User: {question1}")
    print("Agent thinking...")

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": question1}]
    )

    answer1 = response.choices[0].message.content
    print(f"Agent: {answer1}\n")

    # 2. Ask a follow-up question later
    question2 = "Can you remind me what the password is?"
    print(f"User: {question2}")
    print("Agent thinking (and automatically recalling memory!)...")

    response2 = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": question2}]
    )

    answer2 = response2.choices[0].message.content
    print(f"Agent: {answer2}\n")

    # 3. Inspect memory
    print("="*50)
    print("Direct Memory Recall Inspection:")
    hits = client.recall("password", limit=4)
    for i, hit in enumerate(hits, 1):
        print(f"  {i}. [{hit['memory_type']}] {hit['content']}")

if __name__ == "__main__":
    main()
