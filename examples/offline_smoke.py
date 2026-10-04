"""
NSN Three-Line Integration: Offline Smoke Example
Demonstrates persistent memory with zero network calls and zero model downloads.
"""
import nsn

# 1. Initialize memory runtime
nsn.init("./agent-memory")

# 2. Deterministic model fixture simulating an SLM
def simple_slm(prompt: str) -> str:
    print("\n--- Model received prompt ---")
    print(prompt)
    print("-----------------------------\n")
    if "<retrieved_evidence>" in prompt:
        return "I found the required evidence in my memory context above."
    return "Acknowledged, no prior context was retrieved."

# 3. Wrap model with NSN
model = nsn.wrap(simple_slm, namespace="demo_agent")

if __name__ == "__main__":
    print("=== Turn 1: Storing a fact ===")
    r1 = model("Service payment gateway port is configured to 8080.")
    print("Model Output 1:", r1)

    print("\n=== Turn 2: Asking for the fact ===")
    r2 = model("What port does the payment gateway run on?")
    print("Model Output 2:", r2)
