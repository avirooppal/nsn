import json
from neurosleepnet.graph.schemas import GraphNode, GraphEdge
from neurosleepnet.graph.extractor import EntityExtractor, RelationshipExtractor

class GraphBuilder:
    """
    Coordinates entity/relationship extraction and graph storage,
    linking the resulting graph components back to the source memory.
    """
    def __init__(self, storage):
        self.storage = storage
        self.entity_extractor = EntityExtractor()
        self.rel_extractor = RelationshipExtractor(self.entity_extractor)

    def process_memory(self, memory_record: dict):
        relations = self.rel_extractor.extract(memory_record['content'])
        
        node_ids = []
        for rel in relations:
            # We add source_memory link in the properties of nodes and edges
            properties = json.dumps({"source_memory": memory_record['id']})
            
            src_node = GraphNode(label=rel['source']['label'], name=rel['source']['name'])
            self.storage.store_graph_node(src_node.id, src_node.label, src_node.name, properties, src_node.created_at)
            node_ids.append(src_node.id)
            
            tgt_node = GraphNode(label=rel['target']['label'], name=rel['target']['name'])
            self.storage.store_graph_node(tgt_node.id, tgt_node.label, tgt_node.name, properties, tgt_node.created_at)
            node_ids.append(tgt_node.id)
            
            edge = GraphEdge(source_id=src_node.id, target_id=tgt_node.id, relation=rel['relation'])
            self.storage.store_graph_edge(edge.id, edge.source_id, edge.target_id, edge.relation, properties, edge.created_at)
            
        # Update memory metadata with linked graph nodes
        if node_ids:
            meta = memory_record.get('metadata', {})
            meta['linked_graph_nodes'] = node_ids
            self.storage.store(
                memory_id=memory_record['id'],
                content=memory_record['content'],
                created_at=memory_record['created_at'],
                metadata=json.dumps(meta),
                importance=memory_record.get('importance', 0.0),
                trust_score=memory_record.get('trust_score', 0.5),
                embedding=json.dumps(memory_record.get('embedding', []))
            )
