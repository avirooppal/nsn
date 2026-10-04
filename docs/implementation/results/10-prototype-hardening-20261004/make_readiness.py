"""Regenerate prototype gates from immutable measured artifacts."""
import hashlib,json,zipfile
from pathlib import Path
root=Path(__file__).resolve().parent
repo=root.parents[3]
wheel=repo/'dist/prototype-hardening-20261004/nsn-0.3.0-py3-none-any.whl'
minimal=json.loads((root/'minimal-profile.json').read_text())
semantic=json.loads((root/'onnx-profile.json').read_text())
sha=hashlib.sha256(wheel.read_bytes()).hexdigest()
assert minimal['verified_wheel_sha256']==semantic['verified_wheel_sha256']==sha
matrices={name:json.loads((root/(name+'-matrix.json')).read_text()) for name in ('windows','linux')}
assert all(cell['passed'] and cell['wheel_sha256']==sha for cells in matrices.values() for cell in cells.values())
with zipfile.ZipFile(wheel) as z:
 for name in z.namelist():
  if name.startswith(('nsn/','neurosleepnet/')) and not name.endswith('/'):assert z.read(name)==(repo/name).read_bytes(),name
 assert not any(name.startswith(('tests/','benchmarks/','docs/')) for name in z.namelist())
 assert not any(line.startswith(('License:','License-Expression:')) for line in z.read('nsn-0.3.0.dist-info/METADATA').decode().splitlines())
limits={
 'wheel_under_5mb':(wheel.stat().st_size,5*1024*1024),
 'minimal_append_p95_ms':(minimal['append_p95_ms'],10),
 'minimal_repeated_retrieval_p95_ms':(minimal['warm_retrieve_p95_ms'],25),
 'minimal_fresh_retrieval_p95_ms':(minimal['additional_unique_query_p95_ms'],25),
 'minimal_conservative_peak_rss_bytes':(minimal['windows_sum_of_individual_peak_working_sets_bytes'],100*1024*1024),
 'onnx_deferred_append_p95_ms':(semantic['deferred_semantic_append_p95_ms'],10),
 'onnx_repeated_retrieval_p95_ms':(semantic['warm_retrieve_p95_ms'],100),
 'onnx_fresh_retrieval_p95_ms':(semantic['additional_unique_query_p95_ms'],100),
 'onnx_incremental_rss_bytes':(semantic['sampled_incremental_runtime_rss_bytes'],400*1024*1024),
 'onnx_encoder_assets_bytes':(semantic['encoder_assets_bytes'],100*1024*1024)}
record={'status':'hardened_controlled_prototype; full_production_qualification_incomplete','wheel':str(wheel),'wheel_bytes':wheel.stat().st_size,'wheel_sha256':sha,
 'functional_result':(root/'functional.txt').read_text().splitlines()[-1],
 'installed_core_cells':{name:{version:cell['passed'] for version,cell in cells.items()} for name,cells in matrices.items()},
 'hardware_qualification':{'passed':False,'result':(root/'hardware-qualification.txt').read_text().splitlines()[-1],'miss':'1k synthetic vector warm median 10.3357ms exceeds unchanged 10ms assertion'},
 'network_denied_onnx':json.loads((root/'onnx-offline.json').read_text()),
 'memory_only_probe':json.loads((root/'memory-probe.json').read_text())['summary'],
 'resource_gates':{name:{'measured':value,'target_under':target,'passed':value<target} for name,(value,target) in limits.items()},
 'scope':'10k short English events, single host; no answering-model overhead in isolated profiles; repeated/fresh workloads unchanged',
 'open_gates':['minimal and ONNX fresh-query latency','default inline semantic and PyTorch qualification','full frozen public/competitor evaluation and public feature ablations','human calibration of independent grading','hosted CI evidence','owner source redistribution license','100k/1M, large-document, multilingual and service-load qualification'],
 'source_license':'owner choice unresolved','published':False,'new_answer_accuracy_claim':False,
 'notes':['Original wheel/raw artifacts preserved. Only pack.py and runtime.py SDK bytes changed.','Evidence boundary escaping does not guarantee natural-language injection immunity.','Deferred semantic appends qualify foreground durable writes, not inline encoding; replay costs separate.','A higher recall in this small previously observed sample is exploratory, not confirmatory superiority.']}
(root/'readiness.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file() and p.name!='checksums.json'}
(root/'checksums.json').write_text(json.dumps(files,indent=2),encoding='utf-8')
print(json.dumps(record['resource_gates'],indent=2))
