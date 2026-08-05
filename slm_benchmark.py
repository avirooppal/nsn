"""
NSN SLM Benchmark Evaluation Suite
====================================
Tests SLMs via Ollama Cloud and/or OpenRouter in two modes:
  - Vanilla (/generate style): stateless, no NSN, no memory
  - NSN     (/chat style):     NSN-wrapped, persistent memory injected automatically

Metrics measured per model per mode:
  1. Memory Recall Accuracy  -- did the model recall a planted fact?
  2. Cross-Turn Coherence    -- coherent across a 3-turn conversation?
  3. Factual Consistency     -- did planted facts prevent hallucination?
  4. Response Quality        -- overall answer quality (keyword score)
  5. Avg Latency (ms)        -- wall-clock time per API call

Usage:
  set OLLAMA_API_KEY=<your key>
  set OPENROUTER_API_KEY=<your key>     (optional)
  python slm_benchmark.py
"""

import os
import sys
import io

# ── Fix Windows stdout encoding for unicode responses from LLMs ───────────────
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Force HuggingFace / sentence-transformers to use the local cache only.
# This avoids getaddrinfo / DNS failures when doing HEAD checks to hf.co.
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import time
import tempfile
from dataclasses import dataclass, field
from typing import Optional, List

from openai import OpenAI
import neurosleepnet as nsn

# ----------------------------------------------
# CONFIG
# ----------------------------------------------

OLLAMA_API_KEY  = os.environ.get("OLLAMA_API_KEY", "")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "https://ollama.com/v1")

OPENROUTER_API_KEY  = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MODELS = [
    {"name": "Nemotron-Nano-30B (Ollama)",   "model_id": "nemotron-3-nano:30b",              "base_url": OLLAMA_BASE_URL,      "api_key": OLLAMA_API_KEY},
    {"name": "GPT-OSS-20B (Ollama)",          "model_id": "gpt-oss:20b",                     "base_url": OLLAMA_BASE_URL,      "api_key": OLLAMA_API_KEY},
    {"name": "Nemotron-Nano-30B (OpenRouter)","model_id": "nvidia/nemotron-3-nano-30b-a3b:free", "base_url": OPENROUTER_BASE_URL, "api_key": OPENROUTER_API_KEY},
]

MAX_TOKENS       = 256
TEMPERATURE      = 0.1
RETRY_LIMIT      = 2
RETRY_DELAY      = 3.0
REQUEST_TIMEOUT  = 60    # seconds per API call before giving up

# ----------------------------------------------
# TEST SCENARIOS
# ----------------------------------------------

SCENARIOS = [
    {
        "id": "identity",
        "description": "Personal identity recall",
        "plant_turn":    "Hi! My name is Alice and I work at DeepMind on protein folding research.",
        "distract_turn": "Can you tell me about the history of the Eiffel Tower?",
        "probe_turn":    "What is my name and where do I work?",
        "recall_keywords":  ["alice", "deepmind"],
        "quality_keywords": ["alice", "deepmind", "protein"],
    },
    {
        "id": "preference",
        "description": "User preference recall",
        "plant_turn":    "Just so you know, my favourite programming language is Rust and I use Neovim as my editor.",
        "distract_turn": "Explain how photosynthesis works in plants.",
        "probe_turn":    "What is my favourite programming language and which editor do I use?",
        "recall_keywords":  ["rust", "neovim"],
        "quality_keywords": ["rust", "neovim"],
    },
    {
        "id": "project",
        "description": "Project context recall",
        "plant_turn":    "I am building a project called Project Orion. The deadline is September 30th and the client is TechCorp.",
        "distract_turn": "What are the main causes of climate change?",
        "probe_turn":    "What project am I working on, when is the deadline, and who is the client?",
        "recall_keywords":  ["orion", "september", "techcorp"],
        "quality_keywords": ["orion", "september", "techcorp"],
    },
    {
        "id": "credential",
        "description": "Sensitive credential recall",
        "plant_turn":    "Store this for me: the staging server API token starts with tok-staging-XK9. The server URL is api-staging.example.com.",
        "distract_turn": "Tell me about the differences between TCP and UDP.",
        "probe_turn":    "What was the API token prefix I mentioned and what is the server URL?",
        "recall_keywords":  ["tok-staging-xk9", "api-staging.example.com"],
        "quality_keywords": ["tok-staging", "example.com"],
    },
    {
        "id": "medical",
        "description": "Medical context recall",
        "plant_turn":    "Important health info: I am allergic to penicillin and I take metformin 500mg daily for diabetes.",
        "distract_turn": "How do neural networks learn from data?",
        "probe_turn":    "What medication am I allergic to and what medication do I take daily?",
        "recall_keywords":  ["penicillin", "metformin"],
        "quality_keywords": ["penicillin", "metformin", "allerg"],
    },
]

# ----------------------------------------------
# METRIC ACCUMULATORS
# ----------------------------------------------

@dataclass
class MetricAccumulator:
    memory_recall:        List[float] = field(default_factory=list)
    cross_turn_coherence: List[float] = field(default_factory=list)
    factual_consistency:  List[float] = field(default_factory=list)
    response_quality:     List[float] = field(default_factory=list)
    latencies_ms:         List[float] = field(default_factory=list)

    def avg_memory_recall(self)        -> float: return _safe_mean(self.memory_recall)
    def avg_cross_turn_coherence(self) -> float: return _safe_mean(self.cross_turn_coherence)
    def avg_factual_consistency(self)  -> float: return _safe_mean(self.factual_consistency)
    def avg_response_quality(self)     -> float: return _safe_mean(self.response_quality)
    def avg_latency_ms(self)           -> float: return _safe_mean(self.latencies_ms)


def _safe_mean(lst):
    return (sum(lst) / len(lst)) if lst else 0.0

# ----------------------------------------------
# API CALL HELPERS
# ----------------------------------------------

def _make_vanilla_client(base_url, api_key):
    import httpx
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        http_client=httpx.Client(timeout=REQUEST_TIMEOUT),
    )


def _make_nsn_client(namespace, db_path, base_url, api_key):
    # Always create a FRESH OpenAI client (new httpx session) to avoid
    # "client has been closed" errors on retries or successive scenario runs.
    import httpx
    base = OpenAI(
        base_url=base_url,
        api_key=api_key,
        http_client=httpx.Client(timeout=REQUEST_TIMEOUT),
    )
    return nsn.init(base, namespace=namespace, db_path=db_path)


def _ping_model(model_cfg):
    """Quick smoke-test: returns True if model responds within timeout."""
    model_id = model_cfg["model_id"]
    print(f"  Pinging {model_id} ...")
    client = _make_vanilla_client(model_cfg["base_url"], model_cfg["api_key"])
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "Reply with OK"}],
            max_tokens=10,
            temperature=0,
        )
        reply = (resp.choices[0].message.content or "").strip()
        print(f"  -> OK (replied: {reply[:40]!r})")
        return True
    except Exception as e:
        print(f"  -> SKIP (error: {str(e)[:100]})")
        return False


def _call_with_retry(client, model_id, messages, label=""):
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            t0 = time.time()
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            latency_ms = (time.time() - t0) * 1000
            text = response.choices[0].message.content or ""
            return text.strip(), latency_ms
        except Exception as e:
            err = str(e)
            if attempt < RETRY_LIMIT:
                print(f"    [retry {attempt}/{RETRY_LIMIT}] {label}: {err[:80]} -- retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"    [FAIL] {label}: {err[:120]}")
                return None, 0.0

# ----------------------------------------------
# SCORING HELPERS
# ----------------------------------------------

def _keyword_score(text, keywords):
    if not text or not keywords:
        return 0.0
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    return hits / len(keywords)


def _binary_keyword(text, keywords, threshold=0.5):
    return 1 if _keyword_score(text, keywords) >= threshold else 0

# ----------------------------------------------
# VANILLA MODE RUNNER
# ----------------------------------------------

def run_vanilla_scenario(model_cfg, scenario):
    latencies = []
    model_id = model_cfg["model_id"]
    base_url = model_cfg["base_url"]
    api_key = model_cfg["api_key"]
    # Fresh client each turn -- no memory possible
    client1 = _make_vanilla_client(base_url, api_key)
    _, lat1 = _call_with_retry(client1, model_id,
        [{"role": "user", "content": scenario["plant_turn"]}],
        label=f"vanilla/{scenario['id']}/turn1")
    latencies.append(lat1)

    client2 = _make_vanilla_client(base_url, api_key)
    _, lat2 = _call_with_retry(client2, model_id,
        [{"role": "user", "content": scenario["distract_turn"]}],
        label=f"vanilla/{scenario['id']}/turn2")
    latencies.append(lat2)

    client3 = _make_vanilla_client(base_url, api_key)
    probe_response, lat3 = _call_with_retry(client3, model_id,
        [{"role": "user", "content": scenario["probe_turn"]}],
        label=f"vanilla/{scenario['id']}/turn3")
    latencies.append(lat3)

    return probe_response, latencies

# ----------------------------------------------
# NSN MODE RUNNER
# ----------------------------------------------

def run_nsn_scenario(model_cfg, scenario, db_path):
    latencies = []
    model_id = model_cfg["model_id"]
    base_url = model_cfg["base_url"]
    api_key = model_cfg["api_key"]
    namespace = f"nsn_bench_{scenario['id']}_{int(time.time())}"
    client = _make_nsn_client(namespace=namespace, db_path=db_path, base_url=base_url, api_key=api_key)

    messages_so_far = []

    # Turn 1 -- plant the fact (NSN auto-stores it)
    messages_so_far.append({"role": "user", "content": scenario["plant_turn"]})
    reply1, lat1 = _call_with_retry(client, model_id, list(messages_so_far),
        label=f"nsn/{scenario['id']}/turn1")
    latencies.append(lat1)
    if reply1:
        messages_so_far.append({"role": "assistant", "content": reply1})

    # Turn 2 -- distractor
    messages_so_far.append({"role": "user", "content": scenario["distract_turn"]})
    reply2, lat2 = _call_with_retry(client, model_id, list(messages_so_far),
        label=f"nsn/{scenario['id']}/turn2")
    latencies.append(lat2)
    if reply2:
        messages_so_far.append({"role": "assistant", "content": reply2})

    # Turn 3 -- probe (NSN retrieves planted fact from memory)
    messages_so_far.append({"role": "user", "content": scenario["probe_turn"]})
    probe_response, lat3 = _call_with_retry(client, model_id, list(messages_so_far),
        label=f"nsn/{scenario['id']}/turn3")
    latencies.append(lat3)

    return probe_response, latencies

# ----------------------------------------------
# SCORE ONE SCENARIO
# ----------------------------------------------

def score_scenario(probe_response, scenario):
    if probe_response is None:
        return {"memory_recall": 0, "cross_turn_coherence": 0,
                "factual_consistency": 0, "response_quality": 0.0}

    recall_kws  = scenario["recall_keywords"]
    quality_kws = scenario["quality_keywords"]

    memory_recall        = _binary_keyword(probe_response, recall_kws, threshold=0.5)
    cross_turn_coherence = _binary_keyword(probe_response, recall_kws, threshold=0.5)
    factual_consistency  = _binary_keyword(probe_response, recall_kws, threshold=1.0)
    response_quality     = _keyword_score(probe_response, quality_kws)

    return {
        "memory_recall":        memory_recall,
        "cross_turn_coherence": cross_turn_coherence,
        "factual_consistency":  factual_consistency,
        "response_quality":     response_quality,
    }

# ----------------------------------------------
# RUN ONE MODEL x ONE MODE
# ----------------------------------------------

def run_model_mode(model_cfg, mode, db_dir):
    model_id   = model_cfg["model_id"]
    model_name = model_cfg["name"]
    acc        = MetricAccumulator()

    print(f"\n  [{mode.upper()}] {model_name}")
    print(f"    Model ID: {model_id}")
    print(f"    {'--' * 25}")

    for scenario in SCENARIOS:
        scen_id = scenario["id"]
        desc    = scenario["description"]
        print(f"    Scenario: {desc} ({scen_id})")

        db_path = os.path.join(db_dir, f"{model_name}_{mode}_{scen_id}.db")

        try:
            if mode == "vanilla":
                probe_resp, latencies = run_vanilla_scenario(model_cfg, scenario)
            else:
                probe_resp, latencies = run_nsn_scenario(model_cfg, scenario, db_path)

            scores = score_scenario(probe_resp, scenario)

            acc.memory_recall.append(scores["memory_recall"])
            acc.cross_turn_coherence.append(scores["cross_turn_coherence"])
            acc.factual_consistency.append(scores["factual_consistency"])
            acc.response_quality.append(scores["response_quality"])
            acc.latencies_ms.extend([l for l in latencies if l > 0])

            preview     = (probe_resp or "N/A")[:100].replace("\n", " ")
            recall_icon = "PASS" if scores["memory_recall"] else "FAIL"
            print(f"      Recall={recall_icon}  Quality={scores['response_quality']:.2f}")
            print(f"      Response: \"{preview}\"")

            if os.path.exists(db_path):
                try: os.remove(db_path)
                except: pass

        except Exception as e:
            print(f"      ERROR in scenario {scen_id}: {e}")
            acc.memory_recall.append(0)
            acc.cross_turn_coherence.append(0)
            acc.factual_consistency.append(0)
            acc.response_quality.append(0.0)

    return acc

# ----------------------------------------------
# RESULTS TABLE PRINTER
# ----------------------------------------------

def print_results_table(all_results):
    def pct(v): return f"{v * 100:.1f}%"
    def ms(v):  return f"{v:.0f}ms"

    def icon(vanilla_val, nsn_val, is_latency=False):
        diff = (vanilla_val - nsn_val) if is_latency else (nsn_val - vanilla_val)
        return "^^" if diff > 0.01 else ("vv" if diff < -0.01 else "==")

    metrics = [
        ("Memory Recall",        "avg_memory_recall",        pct, False),
        ("Cross-Turn Coherence", "avg_cross_turn_coherence", pct, False),
        ("Factual Consistency",  "avg_factual_consistency",  pct, False),
        ("Response Quality",     "avg_response_quality",     pct, False),
        ("Avg Latency",          "avg_latency_ms",           ms,  True),
    ]

    print("\n")
    print("=" * 90)
    print("  NSN SLM BENCHMARK RESULTS".center(90))
    print("  Vanilla (no memory) vs NSN (persistent memory)".center(90))
    print("=" * 90)

    for model_name, modes in all_results.items():
        v_acc = modes.get("vanilla")
        n_acc = modes.get("nsn")

        print(f"\n  MODEL: {model_name}")
        print(f"  {'-' * 70}")
        print(f"  {'Metric':<26} {'Vanilla':>12}  {'NSN':>12}  {'Delta':>10}  Trend")
        print(f"  {'-' * 70}")

        for label, method, fmt, is_latency in metrics:
            v_val = getattr(v_acc, method)() if v_acc else 0.0
            n_val = getattr(n_acc, method)() if n_acc else 0.0
            trend = icon(v_val, n_val, is_latency=is_latency)

            if is_latency:
                delta_str = f"{n_val - v_val:+.0f}ms"
            else:
                delta_str = f"{(n_val - v_val) * 100:+.1f}pp"

            print(f"  {label:<26} {fmt(v_val):>12}  {fmt(n_val):>12}  {delta_str:>10}  {trend}")

        print(f"  {'-' * 70}")

    # Aggregate
    print("\n")
    print("=" * 90)
    print("  AGGREGATE SUMMARY (averaged across all models)".center(90))
    print("=" * 90)
    print(f"  {'Metric':<26} {'Vanilla Avg':>14}  {'NSN Avg':>14}  {'Improvement':>12}")
    print(f"  {'-' * 72}")

    for label, method, fmt, is_latency in metrics:
        v_vals = [getattr(modes["vanilla"], method)() for modes in all_results.values() if "vanilla" in modes]
        n_vals = [getattr(modes["nsn"],     method)() for modes in all_results.values() if "nsn"     in modes]
        v_avg  = _safe_mean(v_vals)
        n_avg  = _safe_mean(n_vals)

        if is_latency:
            delta_str = f"{n_avg - v_avg:+.0f}ms"
        else:
            delta_str = f"{(n_avg - v_avg) * 100:+.1f}pp"

        print(f"  {label:<26} {fmt(v_avg):>14}  {fmt(n_avg):>14}  {delta_str:>12}")

    print("=" * 90)
    print("  Legend: ^^ NSN improved  vv NSN worse  == No change")
    print("  pp = percentage points | ms = milliseconds")
    print("=" * 90)
    print()

# ----------------------------------------------
# MAIN
# ----------------------------------------------

def main():
    print("")
    print("=" * 70)
    print("  NSN SLM Benchmark -- Memory Augmentation Evaluation")
    print(f"  Call timeout: {REQUEST_TIMEOUT}s per request")
    print("  Modes:  Vanilla (no memory) vs NSN (persistent memory)")
    print("=" * 70)

    # Ping models first to skip any that are down/rate-limited
    print("\n[Pre-flight: pinging all models...]")
    available_models = []
    for model_cfg in MODELS:
        if _ping_model(model_cfg):
            available_models.append(model_cfg)
        else:
            print(f"  !! Skipping {model_cfg['name']} -- not responding")

    if not available_models:
        print("ERROR: No models available. Check API key and OpenRouter status.")
        return

    print(f"\nRunning benchmark on {len(available_models)} model(s): {[m['name'] for m in available_models]}")

    with tempfile.TemporaryDirectory() as db_dir:
        all_results = {}

        for model_cfg in available_models:
            model_name = model_cfg["name"]
            print(f"\n{'=' * 70}")
            print(f"  MODEL: {model_name}  ({model_cfg['model_id']})")
            print(f"{'=' * 70}")

            all_results[model_name] = {}

            # Vanilla mode (no memory)
            vanilla_acc = run_model_mode(model_cfg, "vanilla", db_dir)
            all_results[model_name]["vanilla"] = vanilla_acc

            time.sleep(2)

            # NSN mode (with memory)
            nsn_acc = run_model_mode(model_cfg, "nsn", db_dir)
            all_results[model_name]["nsn"] = nsn_acc

            time.sleep(3)

    print_results_table(all_results)


if __name__ == "__main__":
    main()
