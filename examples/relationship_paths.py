"""Run with python -m examples.relationship_paths. No model or network required."""
import tempfile
import nsn


def main():
    with tempfile.TemporaryDirectory(prefix="nsn_relations_demo_") as directory:
        with nsn.Runtime(directory, graph=True) as memory:
            for statement in ("Alice works for Acme.", "Acme is located in Paris."):
                event_id = memory.append_event(role="user", content=statement)
                memory.extract_relations(event_id)
            evidence = memory.retrieve_pack("Which city hosts Alice's employer?")
            print(evidence.rendered_text)


if __name__ == "__main__":
    main()
