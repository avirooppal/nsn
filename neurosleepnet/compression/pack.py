import json
from neurosleepnet.compression.compressor import ContextCompressor

class ReasoningPackGenerator:
    """
    Generates structured reasoning packs for Small Language Models (SLMs).
    A reasoning pack provides an SLM with highly condensed context and 
    explicit logical rules extracted from the Knowledge Graph.
    """
    def __init__(self, storage, compressor=None):
        self.storage = storage
        self.compressor = compressor or ContextCompressor(max_tokens=200)

    def generate_pack(self, topic: str) -> str:
        """
        Generates a reasoning pack around a specific topic.
        """
        # 1. Retrieve memories relevant to the topic
        memories = self.storage.search_keyword(topic, limit=10)
        
        # 2. Compress the episodic/semantic context
        context = self.compressor.compress(memories, query=topic)
        
        # 3. Extract logical rules from the Knowledge Graph
        graph_data = self.storage.query_graph(topic)
        
        logical_rules = []
        if graph_data and "edges" in graph_data:
            node_name = graph_data['node']['name']
            for edge in graph_data["edges"]:
                relation = edge['relation']
                target_name = edge['target']['name']
                logical_rules.append(f"({node_name}) -[{relation}]-> ({target_name})")

        # 4. Assemble the Reasoning Pack payload
        pack = {
            "topic": topic,
            "context": context,
            "logical_rules": logical_rules,
            "system_prompt": "You are a reasoning engine. Utilize the provided context and logical rules to answer the user's queries precisely."
        }
        
        return json.dumps(pack, indent=2)
