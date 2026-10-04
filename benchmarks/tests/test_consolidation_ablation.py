from benchmarks.run_consolidation_ablation import run


def test_consolidation_ablation_same_originals_and_no_model_work():
    result = run()
    off, on = result["modes"]["sleep_off"], result["modes"]["sleep_on"]
    assert off["original_events"] == on["original_events"] == 9
    assert off["model_calls"] == on["model_calls"] == 0
    # Record a negative ablation honestly, without ranking summaries to favor it.
    assert on["complete_value_rate"] == off["complete_value_rate"] == 0.25
    assert on["mean_gold_source_coverage"] == off["mean_gold_source_coverage"]
    assert on["summaries"] == 4
