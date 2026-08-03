import json
from neurosleepnet.compression.compressor import ContextCompressor


class ReasoningPackGenerator:
    """
    Generates structured reasoning packs for Small Language Models (SLMs).
    A reasoning pack provides an SLM with highly condensed context and
    explicit logical rules extracted from the Knowledge Graph.
    """

    def __init__(self, storage, memory=None, compressor=None):
        self.storage = storage
        self.memory = memory
        self.compressor = compressor or ContextCompressor(max_tokens=200)

    def generate_pack(self, topic: str) -> str:
        """Generates a JSON reasoning pack for SLMs."""
        if self.memory is not None:
            results = self.memory.search_hybrid(topic, limit=10)
        else:
            results = self.storage.search_keyword(topic, limit=10)

        # Relative key_facts: always surface top-3 by importance
        key_facts = sorted(results, key=lambda r: float(r.get('importance', 0)), reverse=True)[:3]

        pack = {
            "topic": topic,
            "context": [r['content'] for r in results],
            "key_facts": [r['content'] for r in key_facts],
            "logical_rules": self._extract_rules(topic=topic),
            "system_prompt": (
                "You are a reasoning engine with access to long-term memory. "
                "Use the provided context, key facts, and logical rules to answer precisely. "
                "Prefer information from key_facts when answering direct questions."
            ),
        }
        return json.dumps(pack)

    def _extract_rules(self, topic=None) -> list:
        """Extracts graph-based logical rules for a given topic."""
        logical_rules = []
        if not topic:
            return logical_rules
        graph_data = self.storage.query_graph(topic)
        if graph_data and "edges" in graph_data:
            node_name = graph_data['node']['name']
            for edge in graph_data["edges"]:
                relation = edge['relation']
                target_name = edge['target']['name']
                logical_rules.append(f"({node_name}) -[{relation}]-> ({target_name})")
        return logical_rules
