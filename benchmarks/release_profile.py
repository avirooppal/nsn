"""Isolated installed-wheel minimal profile; no answering model or pytest RSS."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time
import hashlib
import psutil

CHILD = '''
import json,sqlite3,statistics,sys,time,importlib.metadata
from pathlib import Path
import nsn
assert 'site-packages' in Path(nsn.__file__).parts
n=int(sys.argv[1]);rt=nsn.init('./memory');appends=[]
for i in range(n):
    start=time.perf_counter()
    rt.append_event(namespace='profile',content=f'Service service-{i} gateway port {8000+i}.')
    appends.append((time.perf_counter()-start)*1000)
latencies=[]
for _ in range(40):
    start=time.perf_counter();pack=rt.retrieve_pack(f'service-{n-1} gateway port',namespace='profile')
    latencies.append((time.perf_counter()-start)*1000)
    assert pack.items
with sqlite3.connect(rt.db_path) as conn:
    assert conn.execute("SELECT COUNT(*) FROM events WHERE namespace='profile'").fetchone()[0]==n
unique=[]
for i in range(40):
    start=time.perf_counter();pack=rt.retrieve_pack(f'service-{i} gateway port',namespace='profile')
    unique.append((time.perf_counter()-start)*1000);assert pack.items
disk=sum(p.stat().st_size for p in Path('./memory').glob('*') if p.is_file())
rt.close()
def p95(v):return sorted(v)[int(.95*(len(v)-1))]
print(json.dumps({'records':n,'content':'short generated English service/port records','append_p95_ms':p95(appends),'first_retrieve_ms':latencies[0],'warm_retrieve_p95_ms':p95(latencies[1:]),'additional_unique_query_p95_ms':p95(unique),'disk_bytes_including_wal':disk,'module':nsn.__file__,'installed_distribution':json.loads(importlib.metadata.distribution('nsn').read_text('direct_url.json')),'query_workload':'original same fixed query 40 times; unchanged database; plus 40 separately reported unique queries'}))
'''

SEMANTIC_CHILD = '''
import json,sqlite3,sys,time,importlib.metadata
from pathlib import Path
import psutil
import nsn
assert 'site-packages' in Path(nsn.__file__).parts
baseline=psutil.Process().memory_info().rss
n=int(sys.argv[1]);assets=Path(sys.argv[2]);backend=sys.argv[3]
if backend=='onnx':
    from neurosleepnet.embeddings.onnx_pooled import ONNXPooledEncoder
    encoder=ONNXPooledEncoder(str(assets),cache_size=0)
else:
    import torch
    torch.set_num_threads(1)
    from neurosleepnet.embeddings.pooled import LocalPooledEncoder
    encoder=LocalPooledEncoder(local_asset_path=str(assets),offline=True,revision=assets.name,cache_size=0)
import socket
def denied(*a,**k):raise AssertionError('Unexpected semantic network request')
socket.create_connection=denied;socket.socket.connect=denied
deferred=sys.argv[4]=='true'
rt=nsn.Runtime('./memory',encoder=encoder,defer_indexing=deferred)
appends=[];ids=[];texts=[]
for i in range(n):
    content=f'Service service-{i} gateway port {8000+i}.'
    start=time.perf_counter()
    ids.append(rt.storage.append_event(namespace='profile',content=content))
    appends.append((time.perf_counter()-start)*1000);texts.append(content)
start=time.perf_counter()
for offset in range(0,n,64):
    vectors=encoder.encode_batch(texts[offset:offset+64])
    rt.vector_store.add_batch([(eid,vec,'profile','event') for eid,vec in zip(ids[offset:offset+64],vectors)])
index_ms=(time.perf_counter()-start)*1000
with sqlite3.connect(rt.db_path) as conn:
    assert conn.execute("SELECT COUNT(*) FROM events WHERE namespace='profile'").fetchone()[0]==n
assert rt.vector_store.count('profile','event')==n
latencies=[];stages=[]
for _ in range(40):
    start=time.perf_counter();pack=rt.retrieve_pack(f'service-{n-1} gateway port',namespace='profile')
    latencies.append((time.perf_counter()-start)*1000);stages.append(pack.diagnostic_explain['timings'])
    assert pack.items
unique=[]
for i in range(40):
    start=time.perf_counter();pack=rt.retrieve_pack(f'service-{i} gateway port',namespace='profile')
    unique.append((time.perf_counter()-start)*1000);assert pack.items
disk=sum(p.stat().st_size for p in Path('./memory').glob('*') if p.is_file())
foreground=[]
for i in range(100):
    content=f'Service service-{i} gateway port {8000+i}.'
    start=time.perf_counter();eid=rt.append_event(namespace='foreground',content=content)
    foreground.append((time.perf_counter()-start)*1000)
    if not deferred:assert any(row['id']==eid for row in rt.vector_store.search(encoder.encode(content),namespace='foreground',limit=1))
replay_ms=0
if deferred:
    assert rt.vector_store.count('foreground','event')==0
    replay_start=time.perf_counter();assert rt.process_index_jobs(worker_id='profile',limit=100)==100
    replay_ms=(time.perf_counter()-replay_start)*1000
assert rt.vector_store.count('foreground','event')==100
rt.close()
def p95(v):return sorted(v)[int(.95*(len(v)-1))]
print(json.dumps({'records':n,'vectors':n,'baseline_interpreter_rss_bytes':baseline,'content':'short generated English service/port records','raw_durable_append_p95_ms':p95(appends),'semantic_foreground_append_qualified':p95(foreground)<10,'inline_semantic_append_p95_ms':None if deferred else p95(foreground),'deferred_semantic_append_p95_ms':p95(foreground) if deferred else None,'deferred_indexing':deferred,'foreground_replay_ms':replay_ms,'foreground_measurement':('100 additional scoped appends: durable event/outbox only; vectors verified after explicit separately timed replay' if deferred else '100 additional scoped appends: encoding and durable event/vector/job writes; vectors verified inline'),'encoder_backend':backend,'ingestion':'authoritative raw appends then real neural batch indexing (64); foreground inline timing reported separately','index_ms':index_ms,'first_retrieve_ms':latencies[0],'warm_retrieve_p95_ms':p95(latencies[1:]),'additional_unique_query_p95_ms':p95(unique),'query_embedding_cache_size':0,'timings':stages,'disk_bytes_including_wal':disk,'encoder_assets_bytes':sum(p.stat().st_size for p in assets.rglob('*') if p.is_file()),'encoder_revision':encoder.revision,'module':nsn.__file__,'cpu_inference_threads':1,'network_denied':True,'installed_distribution':json.loads(importlib.metadata.distribution('nsn').read_text('direct_url.json')),'query_workload':'original same fixed query 40 times; unchanged database, neural query cache disabled; plus 40 separately reported unique queries'}))
'''

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--python',required=True);parser.add_argument('--records',type=int,default=10000);parser.add_argument('--output',required=True);parser.add_argument('--embedding-path');parser.add_argument('--wheel');parser.add_argument('--encoder-backend',choices=('torch','onnx'),default='torch');parser.add_argument('--defer-indexing',action='store_true')
    args=parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        wheel_sha=None
        if args.wheel:
            wheel=Path(args.wheel).resolve()
            verify="import nsn,zipfile,sys;from pathlib import Path;root=Path(nsn.__file__).resolve().parent.parent;assert 'site-packages' in root.parts;z=zipfile.ZipFile(sys.argv[1]);[(None if z.read(n)==(root/n).read_bytes() else (_ for _ in ()).throw(AssertionError(n))) for n in z.namelist() if n.startswith(('nsn/','neurosleepnet/')) and not n.endswith('/')];print('installed package matches wheel')"
            checked=subprocess.run([args.python,'-c',verify,str(wheel)],cwd=directory,capture_output=True,text=True,encoding='utf-8')
            if checked.returncode:raise RuntimeError(checked.stderr)
            wheel_sha=hashlib.sha256(wheel.read_bytes()).hexdigest()
        command=[args.python,'-c',SEMANTIC_CHILD if args.embedding_path else CHILD,str(args.records)]
        if args.embedding_path:command.extend([args.embedding_path,args.encoder_backend,str(args.defer_indexing).lower()])
        # Weight-loading progress can exceed pipe capacity; avoid waiting forever
        # for a child blocked on stderr while monitoring its process tree.
        error_log=tempfile.TemporaryFile(mode='w+',encoding='utf-8')
        output_log=tempfile.TemporaryFile(mode='w+',encoding='utf-8')
        process=subprocess.Popen(command,cwd=directory,stdout=output_log,stderr=error_log,text=True)
        tracked=psutil.Process(process.pid);samples=[];peaks={}
        while process.poll() is None:
            try:
                rss=0
                for member in [tracked]+tracked.children(recursive=True):
                    try:
                        memory=member.memory_info();rss+=memory.rss
                        peaks[member.pid]=max(peaks.get(member.pid,0),getattr(memory,'peak_wset',0))
                    except psutil.NoSuchProcess:pass
                samples.append(rss)
            except psutil.NoSuchProcess:break
            time.sleep(.02)
        process.wait();output_log.seek(0);stdout=output_log.read();output_log.close()
        error_log.seek(0);stderr=error_log.read();error_log.close()
        if process.returncode:raise RuntimeError(stderr)
    result=json.loads(stdout)
    result['verified_wheel_sha256']=wheel_sha
    result.update({'sampled_process_tree_rss_max_bytes':max(samples,default=0),'windows_sum_of_individual_peak_working_sets_bytes':sum(peaks.values()) or None,'sampling_interval_ms':20,'profile':'isolated minimal installed wheel; launcher and child interpreter process tree','first_query_is_not_os_cache_cold':True,'answering_model':'none','targets':{'warm_retrieve_p95_ms':25,'append_p95_ms':10,'incremental_runtime_rss_bytes':100*1024**2},'rss_gate_interpretation':'sum of OS individual peaks is a conservative bound, not simultaneous peak; no encoder/model loaded'})
    if args.embedding_path:
        result.update(profile='isolated semantic installed wheel; encoder included, no answering model',
                      targets={'warm_retrieve_p95_ms':100,'incremental_runtime_rss_bytes':400*1024**2,'encoder_assets_bytes':100*1024**2},
                      rss_gate_interpretation='sampled simultaneous process-tree peak minus pre-encoder interpreter baseline; OS sum of peaks retained separately')
        result['sampled_incremental_runtime_rss_bytes']=max(samples,default=0)-result['baseline_interpreter_rss_bytes']
    Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
