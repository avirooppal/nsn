"""
NSN Three-Line Integration: Local Ollama / vLLM Example
Connects to a local OpenAI-compatible endpoint (e.g. Ollama at localhost:11434/v1).
"""
import nsn

# 1. Initialize self-hosted memory layer
nsn.init("./agent-memory")

def run_local_agent():
    try:
        from openai import OpenAI
    except ImportError:
        print("Please install openai client: pip install openai")
        return

    # Point client to local Ollama server
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama", # placeholder for local service
    )

    # 2. Wrap client with NSN
    agent = nsn.wrap(client, namespace="ollama_assistant")

    # 3. Interacting with local model automatically injects recalled context
    print("Calling local model with memory...")
    response = agent.chat.completions.create(
        model="llama3.2:1b",
        messages=[
            {"role": "user", "content": "What port does the payment gateway run on?"}
        ]
    )
    print("Agent Response:", response.choices[0].message.content)

if __name__ == "__main__":
    run_local_agent()
