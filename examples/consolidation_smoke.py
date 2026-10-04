"""Run with python -m examples.consolidation_smoke; no network/model needed."""
import tempfile
import nsn


def main():
    with tempfile.TemporaryDirectory(prefix="nsn_sleep_demo_") as directory:
        with nsn.Runtime(directory) as memory:
            memory.append_event(content="Service payment port 8080, timeout 7 ms, deployment uuid-42.")
            report = memory.sleep()
            print(report)
            print(memory.retrieve_pack("payment port", use_summaries=True).rendered_text)
            print("Repeat processed:", memory.sleep()["processed"])


if __name__ == "__main__":
    main()
