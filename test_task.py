from neurosleepnet import Memory
import os

if os.path.exists('neurosleepnet.db'):
    try:
        os.remove('neurosleepnet.db')
    except Exception:
        pass

memory = Memory()

memory.store("The weather in New York is sunny and warm.")
memory.store("Quantum computing uses qubits.")
memory.store("The Golden Gate Bridge is in San Francisco.")

results = memory.search("Where is the famous red bridge?")
print("Search results:")
for r in results:
    print(f"- {r['content']} (Score: {r['score']:.4f})")
