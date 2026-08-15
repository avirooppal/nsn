"""
Gold Evidence dataset schema and large synthetic dataset generator (10,000+ benchmark items).
Every benchmark query explicitly stores gold ground truth IDs, entities, edges, and answers upfront.
"""

import random

class BenchmarkItem:
    def __init__(
        self,
        query_id: str,
        question: str,
        ground_truth_answer: str,
        gold_memory_ids: list,
        gold_entities: list = None,
        gold_edges: list = None,
        gold_temporal_interval: str = None,
        gold_memory_type: str = None,
        gold_namespace: str = "default",
        gold_source: str = "system",
        gold_current_state: str = None,
        gold_historical_state: str = None,
        category: str = "general",
        difficulty: str = "medium"
    ):
        self.query_id = query_id
        self.question = question
        self.ground_truth_answer = ground_truth_answer
        self.gold_memory_ids = gold_memory_ids
        self.gold_entities = gold_entities or []
        self.gold_edges = gold_edges or []
        self.gold_temporal_interval = gold_temporal_interval
        self.gold_memory_type = gold_memory_type
        self.gold_namespace = gold_namespace
        self.gold_source = gold_source
        self.gold_current_state = gold_current_state
        self.gold_historical_state = gold_historical_state
        self.category = category
        self.difficulty = difficulty

    def to_dict(self):
        return {
            "query_id": self.query_id,
            "question": self.question,
            "ground_truth_answer": self.ground_truth_answer,
            "gold_memory_ids": self.gold_memory_ids,
            "gold_entities": self.gold_entities,
            "gold_edges": self.gold_edges,
            "gold_temporal_interval": self.gold_temporal_interval,
            "gold_memory_type": self.gold_memory_type,
            "gold_namespace": self.gold_namespace,
            "gold_source": self.gold_source,
            "gold_current_state": self.gold_current_state,
            "gold_historical_state": self.gold_historical_state,
            "category": self.category,
            "difficulty": self.difficulty
        }

class LargeSyntheticDatasetGenerator:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)

    def generate_gold_update_dataset(self, num_questions: int = 1000):
        items = []
        services = ["PostgreSQL", "Redis", "Kafka", "Elasticsearch", "Nginx", "MongoDB", "RabbitMQ", "ClickHouse", "Cassandra", "Memcached"]
        
        for i in range(num_questions):
            service = services[i % len(services)]
            port_orig = 5000 + (i * 3) % 1000
            port_mid = 6000 + (i * 3) % 1000
            port_curr = 9000 + (i * 3) % 1000

            mem_id_orig = f"mem_upd_{i}_day1"
            mem_id_mid = f"mem_upd_{i}_day10"
            mem_id_curr = f"mem_upd_{i}_day20"

            observations = [
                {"id": mem_id_orig, "content": f"Day 1: {service} production port is configured to {port_orig}.", "timestamp": "Day 1", "source": "system"},
                {"id": mem_id_mid, "content": f"Day 10: {service} production port updated to {port_mid}.", "timestamp": "Day 10", "source": "system"},
                {"id": mem_id_curr, "content": f"Day 20: {service} production port migrated to {port_curr}.", "timestamp": "Day 20", "source": "system"},
            ]

            # Current state question
            q_curr = BenchmarkItem(
                query_id=f"q_upd_{i}_curr",
                question=f"What is the current production port for {service}?",
                ground_truth_answer=str(port_curr),
                gold_memory_ids=[mem_id_curr],
                gold_current_state=str(port_curr),
                gold_historical_state=str(port_orig),
                category="knowledge_update",
                difficulty="medium"
            )

            # Original state question
            q_orig = BenchmarkItem(
                query_id=f"q_upd_{i}_orig",
                question=f"What was the original {service} port on Day 1?",
                ground_truth_answer=str(port_orig),
                gold_memory_ids=[mem_id_orig],
                gold_current_state=str(port_curr),
                gold_historical_state=str(port_orig),
                category="knowledge_update",
                difficulty="medium"
            )

            items.append({"observations": observations, "queries": [q_curr, q_orig]})
            
        return items

    def generate_gold_contradiction_dataset(self, num_questions: int = 1000):
        items = []
        facts = [
            ("database host", "db-primary.prod.net", "db-backup.prod.net"),
            ("admin user", "admin_alice", "admin_bob"),
            ("deployment region", "us-east-1", "eu-west-1"),
            ("ssl cert", "cert_v1.pem", "cert_v2.pem")
        ]
        
        for i in range(num_questions):
            attr, val_low, val_high = facts[i % len(facts)]
            id_low = f"mem_contra_{i}_user"
            id_high = f"mem_contra_{i}_sys"

            observations = [
                {"id": id_low, "content": f"The system {attr} is {val_low}.", "source": "user", "timestamp": "2026-01-01T00:00:00Z"},
                {"id": id_high, "content": f"The system {attr} is not {val_low}. It is {val_high}.", "source": "system", "timestamp": "2026-01-02T00:00:00Z"},
            ]

            q = BenchmarkItem(
                query_id=f"q_contra_{i}",
                question=f"What is the system {attr}?",
                ground_truth_answer=val_high,
                gold_memory_ids=[id_high],
                gold_source="system",
                category="contradiction",
                difficulty="medium"
            )
            items.append({"observations": observations, "query": q})
            
        return items

    def generate_gold_graph_multihop_dataset(self, num_chains: int = 200, max_hops: int = 10):
        items = []
        entities = ["Alice", "Bob", "Project X", "PostgreSQL", "Server 7", "Frankfurt", "Rack 12", "Switch 4", "Gateway 9", "Cluster Zero", "Node Beta"]
        
        for c in range(num_chains):
            chain_obs = []
            chain_edges = []
            for i in range(min(max_hops, len(entities) - 1)):
                mem_id = f"mem_graph_c{c}_h{i+1}"
                src = entities[i]
                tgt = entities[i+1]
                chain_obs.append({"id": mem_id, "content": f"{src} connects to {tgt}."})
                chain_edges.append((src, "connects_to", tgt))

            queries = []
            for h in [1, 2, 3, 5, 10]:
                if h < len(entities):
                    gold_path = entities[:h+1]
                    gold_mems = [f"mem_graph_c{c}_h{k+1}" for k in range(h)]
                    queries.append(BenchmarkItem(
                        query_id=f"q_graph_c{c}_hop{h}",
                        question=f"Starting from {entities[0]} with {h} hop(s), what entity is reached?",
                        ground_truth_answer=entities[h],
                        gold_memory_ids=gold_mems,
                        gold_entities=gold_path,
                        gold_edges=[f"{gold_path[k]}->{gold_path[k+1]}" for k in range(h)],
                        category="multi_hop",
                        difficulty=f"hop_{h}"
                    ))
        return items

    def generate_knowledge_update_test(self, num_sequences: int = 1000):
        return self.generate_gold_update_dataset(num_questions=num_sequences)

    def generate_contradiction_test(self, num_samples: int = 1000):
        return self.generate_gold_contradiction_dataset(num_samples=num_samples)

    def generate_multihop_test(self, num_chains: int = 1000):
        return self.generate_gold_multihop_dataset(num_chains=num_chains)


SyntheticBenchmarkGenerator = LargeSyntheticDatasetGenerator

