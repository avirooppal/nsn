import os
import sys
import time
from openai import OpenAI
import neurosleepnet as nsn

def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "<OLLAMA_CLOUD_API_KEY>")
    OLLAMA_BASE_URL = "https://ollama.com/v1"
    MODELS = ["nemotron-3-nano:30b", "gpt-oss:20b"]

    base_client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)

    # 5 test facts and queries to score numerical accuracy (%)
    test_suite = [
        ("My favorite drink is Iced Vanilla Oat Latte.", "What is my favorite drink?", ["vanilla", "oat", "latte"]),
        ("My backend server port is 8080.", "What is my backend server port?", ["8080"]),
        ("My secret project code name is Project-Hyperion.", "What is my secret project code name?", ["hyperion"]),
        ("My favorite programming language is Python 3.11.", "What is my favorite programming language?", ["python"]),
        ("My database host IP is 192.168.1.150.", "What is my database host IP?", ["192.168.1.150"])
    ]

    results = []

    for model in MODELS:
        print(f"\n==================================================")
        print(f" TESTING MODEL: {model}")
        print(f"==================================================")

        # ---------------------------------------------------------------------
        # 1. NON-NSN (STATELESS BENCHMARK)
        # ---------------------------------------------------------------------
        print("\n--- [1/2] NON-NSN (Stateless / Pure Ollama Cloud API) ---")
        non_nsn_correct = 0
        non_nsn_latencies = []

        # Store all facts (stateless calls)
        for fact, _, _ in test_suite:
            base_client.chat.completions.create(model=model, messages=[{"role": "user", "content": fact}])

        # Ask queries
        for fact, query, expected_keywords in test_suite:
            t0 = time.time()
            res = base_client.chat.completions.create(model=model, messages=[{"role": "user", "content": query}])
            lat = (time.time() - t0) * 1000
            non_nsn_latencies.append(lat)

            content = res.choices[0].message.content.lower()
            hit = any(kw.lower() in content for kw in expected_keywords)
            if hit:
                non_nsn_correct += 1

        non_nsn_acc = (non_nsn_correct / len(test_suite)) * 100
        non_nsn_avg_lat = sum(non_nsn_latencies) / len(non_nsn_latencies)

        # ---------------------------------------------------------------------
        # 2. NSN (NEUROSLEEPNET MEMORY LAYER)
        # ---------------------------------------------------------------------
        print("\n--- [2/2] NSN (With NeuroSleepNet Memory Layer) ---")
        db_path = f"num_bench_{model.replace(':', '_')}.db"
        if os.path.exists(db_path):
            os.remove(db_path)

        nsn_client = nsn.init(base_client, namespace=f"num_bench_{model}", db_path=db_path, recall_limit=5)
        
        nsn_correct = 0
        nsn_latencies = []

        # Feed facts into NSN
        for fact, _, _ in test_suite:
            nsn_client.chat.completions.create(model=model, messages=[{"role": "user", "content": fact}])

        # Ask queries via NSN
        for fact, query, expected_keywords in test_suite:
            t0 = time.time()
            res = nsn_client.chat.completions.create(model=model, messages=[{"role": "user", "content": query}])
            lat = (time.time() - t0) * 1000
            nsn_latencies.append(lat)

            content = res.choices[0].message.content.lower()
            hit = any(kw.lower() in content for kw in expected_keywords)
            if hit:
                nsn_correct += 1

        nsn_acc = (nsn_correct / len(test_suite)) * 100
        nsn_avg_lat = sum(nsn_latencies) / len(nsn_latencies)

        results.append({
            "model": model,
            "non_nsn_acc": f"{non_nsn_acc:.1f}%",
            "non_nsn_score": f"{non_nsn_correct}/{len(test_suite)}",
            "non_nsn_lat": f"{non_nsn_avg_lat:.1f} ms",
            "nsn_acc": f"{nsn_acc:.1f}%",
            "nsn_score": f"{nsn_correct}/{len(test_suite)}",
            "nsn_lat": f"{nsn_avg_lat:.1f} ms",
        })

    # Print Numerical Chart
    print("\n" + "="*95)
    print("                NUMERICAL BENCHMARK POC SUMMARY (5-FACT ACCURACY & LATENCY)")
    print("="*95)
    print(f"{'Model Name':<22} | {'Non-NSN Acc':<12} | {'Non-NSN Lat':<12} | {'NSN Acc':<10} | {'NSN Lat':<12} | {'Recall Gain':<11}")
    print("-" * 95)
    for r in results:
        gain = "+100.0%" if r['nsn_acc'] == "100.0%" and r['non_nsn_acc'] == "0.0%" else "N/A"
        print(f"{r['model']:<22} | {r['non_nsn_score']} ({r['non_nsn_acc']:<5}) | {r['non_nsn_lat']:<12} | {r['nsn_score']} ({r['nsn_acc']:<4}) | {r['nsn_lat']:<12} | {gain:<11}")
    print("="*95 + "\n")

if __name__ == "__main__":
    main()
