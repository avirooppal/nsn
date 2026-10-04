"""Small deterministic host-executed task fixture. No LLM quality claim.

Stored text is never evaluated as code. Driver operations are hard-coded here.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from nsn import Runtime

def run():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        rt = Runtime(str(root/'memory'))
        steps = ['Read UTF-8 file','Decode JSON','Return count of items']
        pid = rt.record_procedure(title='Count JSON items',steps=steps,prerequisites=['Readable input file'],environment={'format':'json'},tools=['file.read','json.decode'])
        training_file = root/'training.json'
        training_file.write_text('[1,2]',encoding='utf-8')
        training_count = len(json.loads(training_file.read_text(encoding='utf-8')))
        assert training_count==2
        outcome = rt.append_event(role='tool',content=f'Training task returned count {training_count}; expected 2')
        rt.record_outcome(procedure_id=pid,event_id=outcome,outcome='success',evidence='observed')
        cases = [('array','[1,2,3]',3),('empty','[]',0),('invalid','not JSON',None)]
        results = []
        for name,content,expected in cases:
            path = root/(name+'.json')
            path.write_text(content,encoding='utf-8')
            procedures = rt.retrieve_procedures('Count JSON items',environment={'format':'json'},tools=['file.read','json.decode'])
            # Explicit authorized fixture driver, not a generic procedure executor.
            try:
                result = len(json.loads(path.read_text(encoding='utf-8')))
                success = result == expected
            except ValueError:
                result,success = None,False
            trace = rt.append_event(role='tool',content=json.dumps({'task':name,'result':result,'expected':expected}),turn_status='completed' if success else 'failed')
            rt.record_outcome(procedure_id=pid,event_id=trace,outcome='success' if success else 'failure',evidence='observed')
            results.append({'task':name,'retrieved':bool(procedures),'success':success,'result':result,'expected':expected})
        report = {'fixture':'json-count-v1','model_calls':0,'tasks':results,'successes':sum(r['success'] for r in results),'total':len(results),'failure_warning_preserved':rt.retrieve_procedures('Count JSON items',environment={'format':'json'},tools=['file.read','json.decode'])[0]['warning'],'claim':'Driver integration evidence only; no baseline or LLM improvement measurement'}
        rt.close()
        return report

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output')
    args = parser.parse_args()
    report = run()
    if args.output:
        Path(args.output).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
