from benchmarks.run_relationship_ablation import run


def test_matched_ablation_rewards_complete_raw_evidence_and_scores_all_queries():
    result = run()
    assert len(result["cases"]) == 16  # All eight queries in both modes.
    by_case = {(r["mode"], r["id"]): r for r in result["cases"]}
    # Raw BM25 evidence can score a complete path without a graph path object.
    assert by_case[("graph_off", "two_hop_employer")]["complete_path"]
    assert by_case[("graph_on", "three_hop")]["complete_path"]
    assert not by_case[("graph_off", "three_hop")]["complete_path"]
    for mode in ("graph_on", "graph_off"):
        assert result["summary"][mode]["direct_recall"]
        assert result["summary"][mode]["unsupported_full_paths"] == 0
