"""Try NSN offline, or with an existing local Ollama model. Uses temporary data."""
import argparse
import json
import tempfile
from urllib.request import Request, urlopen

import nsn


FACT = "Kestrel-X7422 access token is violet-7812."
QUESTION = "What is the Kestrel-X7422 access token?"
TOKEN = "violet-7812"


def offline_model(prompt):
    """Deterministic fixture: verifies evidence delivery, not model intelligence."""
    return TOKEN if "<retrieved_evidence>" in prompt and TOKEN in prompt else "No prior memory."


def ollama_model(name):
    def answer(prompt):
        payload = {
            "model": name, "stream": False, "think": False,
            "messages": [
                {"role": "system", "content": "Use supplied evidence. Answer only the requested token; if absent, answer UNKNOWN."},
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": 0, "seed": 42, "num_ctx": 2048, "num_predict": 32},
        }
        request = Request("http://127.0.0.1:11434/api/chat",
                          data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=120) as response:
            result = json.load(response)
        if not result.get("done"):
            raise RuntimeError("Ollama did not complete the response")
        return result["message"]["content"]
    return answer


def run_demo(answer=offline_model):
    if nsn.get_default_runtime() is not None:
        raise RuntimeError("Close the existing default runtime before running this demo")
    prompts = []

    def captured_model(prompt):
        prompts.append(prompt)
        return answer(prompt)

    with tempfile.TemporaryDirectory(prefix="nsn-try-") as directory:
        try:
            nsn.init(directory)
            model = nsn.wrap(captured_model, namespace="alice")
            model(FACT)
            if "<retrieved_evidence>" in prompts[0]:
                raise AssertionError("First turn recalled itself")
            recalled = model(QUESTION)
            if "<retrieved_evidence>" not in prompts[1] or FACT not in prompts[1]:
                raise AssertionError("Stored fact was not delivered to the model")
            runtime = nsn.get_default_runtime()
            if runtime.retrieve_pack(QUESTION, namespace="bob").items:
                raise AssertionError("Memory leaked into another namespace")
            nsn.close()
            nsn.init(directory)
            model = nsn.wrap(captured_model, namespace="alice")
            restarted = model(QUESTION)
            if "<retrieved_evidence>" not in prompts[2] or FACT not in prompts[2]:
                raise AssertionError("Stored fact was lost after restart")
            if len(prompts) != 3:
                raise AssertionError("Wrapper made unexpected extra model calls")
            return {"recall": recalled, "after_restart": restarted,
                    "memory_checks": ["no self-recall", "prior evidence delivered",
                                      "namespace isolation", "restart persistence", "one model call per turn"]}
        finally:
            nsn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Existing local Ollama model, e.g. qwen3:1.7b; omit for offline fixture")
    args = parser.parse_args()
    print("Backend:", args.model or "offline deterministic fixture (not an LLM)")
    result = run_demo(ollama_model(args.model) if args.model else offline_model)
    for check in result["memory_checks"]:
        print("PASS:", check)
    print("Answer:", result["recall"])
    print("After restart:", result["after_restart"])
    if args.model:
        matched = all(TOKEN in result[key] for key in ("recall", "after_restart"))
        print("Model token-answer check:", "PASS" if matched else "FAIL (memory delivery passed)")
        if not matched:
            raise SystemExit(1)
    print("Temporary memory cleaned up. Existing databases were not modified.")


if __name__ == "__main__":
    main()
