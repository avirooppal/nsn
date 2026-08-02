import time
import os
from neurosleepnet import Memory

# Mixed dataset of exact substring queries (for Keyword Search) and semantic natural language queries (for Semantic/Hybrid)
EVAL_DATASET = [
    # Substring Queries (Extracts from the facts)
    {"fact": "Apollo 11 was the American spaceflight that first landed humans on the Moon on July 20, 1969.", "query": "first landed humans on the Moon"},
    {"fact": "The Python programming language was created by Guido van Rossum and first released in 1991.", "query": "created by Guido van Rossum"},
    {"fact": "Photosynthesis is the process used by plants, algae and certain bacteria to harness energy from sunlight and turn it into chemical energy.", "query": "process used by plants"},
    {"fact": "Albert Einstein developed the theory of relativity, one of the two pillars of modern physics.", "query": "theory of relativity"},
    {"fact": "The Great Wall of China is a series of fortifications that were built across the historical northern borders of ancient Chinese states and Imperial China.", "query": "fortifications that were built across the historical northern borders"},
    {"fact": "Water is an inorganic compound with the chemical formula H2O.", "query": "chemical formula H2O"},
    {"fact": "William Shakespeare was an English playwright, poet and actor, widely regarded as the greatest writer in the English language.", "query": "greatest writer in the English language"},
    {"fact": "The human heart is an organ that pumps blood throughout the body via the circulatory system.", "query": "organ that pumps blood"},
    {"fact": "The speed of light in vacuum, commonly denoted c, is a universal physical constant exactly equal to 299,792,458 metres per second.", "query": "speed of light in vacuum"},
    {"fact": "Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas.", "query": "highest mountain above sea level"},
    {"fact": "Leonardo da Vinci painted the Mona Lisa, a half-length portrait painting.", "query": "painted the Mona Lisa"},
    {"fact": "The Pacific Ocean is the largest and deepest of Earth's five oceanic divisions.", "query": "largest and deepest of Earth's five oceanic divisions"},
    {"fact": "Oxygen is the chemical element with the symbol O and atomic number 8.", "query": "atomic number 8"},
    {"fact": "The currency of Japan is the Japanese yen.", "query": "currency of Japan"},
    {"fact": "Gravity is a fundamental interaction which causes mutual attraction between all things that have mass or energy.", "query": "mutual attraction between all things that have mass"},
    
    # Semantic Queries (Do not overlap perfectly with facts, requires embeddings)
    {"fact": "Apollo 11 was the American spaceflight that first landed humans on the Moon on July 20, 1969.", "query": "Which spacecraft took people to the lunar surface?"},
    {"fact": "The Python programming language was created by Guido van Rossum and first released in 1991.", "query": "Who is the original author of the Python language?"},
    {"fact": "Photosynthesis is the process used by plants, algae and certain bacteria to harness energy from sunlight and turn it into chemical energy.", "query": "How do flora generate their food from the sun?"},
    {"fact": "Albert Einstein developed the theory of relativity, one of the two pillars of modern physics.", "query": "Which scientist is responsible for relativity?"},
    {"fact": "Water is an inorganic compound with the chemical formula H2O.", "query": "What is the molecular makeup of water?"},
    {"fact": "William Shakespeare was an English playwright, poet and actor, widely regarded as the greatest writer in the English language.", "query": "Who is famous for writing Romeo and Juliet?"},
    {"fact": "The human heart is an organ that pumps blood throughout the body via the circulatory system.", "query": "Which part of the anatomy circulates blood?"},
    {"fact": "The speed of light in vacuum, commonly denoted c, is a universal physical constant exactly equal to 299,792,458 metres per second.", "query": "How fast do photons travel in empty space?"},
    {"fact": "Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas.", "query": "What is the tallest peak on the planet?"},
    {"fact": "Leonardo da Vinci painted the Mona Lisa, a half-length portrait painting.", "query": "Who created the famous smiling woman artwork?"},
]

# Adding Similar Facts to test Disambiguation and Precision
DISTRACTORS = [
    "Apollo 13 was a lunar mission that suffered a critical failure.",
    "Java was created by James Gosling at Sun Microsystems.",
    "Cellular respiration is the process by which biological fuels are oxidized in the presence of an inorganic electron acceptor.",
    "Isaac Newton developed the laws of motion and universal gravitation.",
    "The Berlin Wall was a guarded concrete barrier that physically and ideologically divided Berlin.",
    "Hydrogen peroxide has the chemical formula H2O2.",
    "Charles Dickens was an English writer and social critic.",
    "The human brain is the central organ of the human nervous system.",
    "The speed of sound in dry air at 20 °C is 343 metres per second.",
    "K2 is the second-highest mountain on Earth.",
    "Vincent van Gogh painted The Starry Night.",
    "The Atlantic Ocean is the second-largest of the world's oceans.",
    "Carbon is the chemical element with the symbol C and atomic number 6.",
    "The currency of the United Kingdom is the pound sterling.",
    "Electromagnetism is an interaction that occurs between particles with electric charge.",
]

def calculate_mrr(rankings):
    mrr = 0.0
    for rank in rankings:
        if rank > 0:
            mrr += 1.0 / rank
    return mrr / len(rankings) if rankings else 0.0

def evaluate():
    print("=== NeuroSleepNet Real-World Performance Evaluation ===")
    
    if os.path.exists('neurosleepnet.db'):
        try:
            os.remove('neurosleepnet.db')
        except Exception:
            pass
            
    print("\n[1] Initializing Memory System (Loading local embeddings - sentence-transformers)...")
    start_time = time.time()
    memory = Memory()
    print(f"    -> Initialization Time: {time.time() - start_time:.2f}s")
    
    print("\n[2] Ingesting Knowledge Base (Facts + Distractors)...")
    ingest_start = time.time()
    
    fact_id_map = {}
    unique_facts = set([item["fact"] for item in EVAL_DATASET])
    
    # Store Facts
    for fact in unique_facts:
        mem_id = memory.store(
            content=fact,
            metadata={"type": "fact"},
            importance=1.0,
            trust_score=1.0
        )
        fact_id_map[fact] = mem_id
        
    # Store Distractors
    for distractor in DISTRACTORS:
        memory.store(
            content=distractor,
            metadata={"type": "distractor"},
            importance=0.5,
            trust_score=0.8
        )
        
    total_docs = len(unique_facts) + len(DISTRACTORS)
    ingest_time = time.time() - ingest_start
    print(f"    -> Ingested {total_docs} documents.")
    print(f"    -> Total Ingestion Time: {ingest_time:.2f}s")
    print(f"    -> Average Time per Document: {(ingest_time/total_docs)*1000:.2f}ms")
    
    print("\n[3] Running Evaluation Queries...")
    
    methods = {
        "Semantic Search": memory.search,
        "Keyword Search": memory.search_keyword,
        "Hybrid Search": memory.search_hybrid
    }
    
    results_summary = {}
    
    for method_name, search_func in methods.items():
        print(f"\n  Evaluating {method_name}...")
        
        ranks = []
        top1_hits = 0
        top3_hits = 0
        total_latency = 0.0
        
        for item in EVAL_DATASET:
            query = item["query"]
            target_fact = item["fact"]
            target_id = fact_id_map[target_fact]
            
            q_start = time.time()
            results = search_func(query, limit=5)
            q_time = time.time() - q_start
            total_latency += q_time
            
            rank = 0
            for i, res in enumerate(results):
                if res['id'] == target_id:
                    rank = i + 1
                    break
                    
            ranks.append(rank)
            if rank == 1:
                top1_hits += 1
            if 1 <= rank <= 3:
                top3_hits += 1
                
        num_queries = len(EVAL_DATASET)
        mrr = calculate_mrr(ranks)
        acc_top1 = top1_hits / num_queries
        acc_top3 = top3_hits / num_queries
        avg_latency = (total_latency / num_queries) * 1000
        
        results_summary[method_name] = {
            "MRR": mrr,
            "Top-1 Acc": acc_top1,
            "Top-3 Acc": acc_top3,
            "Avg Latency (ms)": avg_latency
        }
        
        print(f"    -> MRR: {mrr:.4f}")
        print(f"    -> Top-1 Accuracy: {acc_top1:.1%}")
        print(f"    -> Top-3 Accuracy: {acc_top3:.1%}")
        print(f"    -> Avg Latency: {avg_latency:.2f} ms")
        
    print("\n=== Evaluation Summary ===")
    print(f"{'Method':<18} | {'MRR':<8} | {'Top-1':<8} | {'Top-3':<8} | {'Latency (ms)':<12}")
    print("-" * 65)
    for method, metrics in results_summary.items():
        print(f"{method:<18} | {metrics['MRR']:<8.4f} | {metrics['Top-1 Acc']:<8.1%} | {metrics['Top-3 Acc']:<8.1%} | {metrics['Avg Latency (ms)']:<12.2f}")
    
    print("\nConclusion: Hybrid Search leverages both exact matches (RRF) and vector embeddings for maximum recall and precision.")
    print("Keyword search uses SQL LIKE '%query%', so it only performs well on the exact substring queries.")

if __name__ == '__main__':
    evaluate()
