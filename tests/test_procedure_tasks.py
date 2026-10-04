from benchmarks.run_procedure_tasks import run

def test_host_executed_tasks_reuse_and_failure_warning():
    report = run()
    assert report['successes']==2 and report['total']==3
    assert all(t['retrieved'] for t in report['tasks'])
    assert report['failure_warning_preserved'] and report['model_calls']==0
