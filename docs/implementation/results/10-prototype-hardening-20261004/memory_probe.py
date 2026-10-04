"""Compare memory evidence before/after fixes; no answering or judge model."""
from contextlib import closing
import hashlib,importlib.util,json,sqlite3,sys,tempfile,statistics
from pathlib import Path
import numpy as np
from nsn import Runtime
from neurosleepnet.retrieval.pack import ContextPacker
from neurosleepnet.storage.compact_vector import CompactVectorStore
from neurosleepnet.embeddings.onnx_pooled import ONNXPooledEncoder
from benchmarks.public_eval.jsonl import read_rows
from benchmarks.public_eval.runner import packed
root=Path(__file__).resolve().parent
parent=root.parent/'09-nsn-vs-no-nsn-20261004/qwen2b'
manifest=json.loads((parent/'manifest.json').read_text())
source=parent/'source_snapshot'
# The SDK snapshot is frozen by the original wheel; obtain its unchanged packer
# from the original installed wheel, not a reconstructed approximation.
import zipfile
oldwheel=Path('dist/nsn-0.3.0-py3-none-any.whl')
assert hashlib.sha256(oldwheel.read_bytes()).hexdigest()=='ce32c7573366aa9d706889b042224f2d974edf2197e17aed34dac8fcc79e1431'
with zipfile.ZipFile(oldwheel) as z:oldsource=z.read('neurosleepnet/retrieval/pack.py')
assert hashlib.sha256(oldsource).hexdigest()==manifest['memory_layer_source_sha256']['neurosleepnet/retrieval/pack.py']
import types
mod=types.ModuleType('nsn_old_packer');sys.modules[mod.__name__]=mod
exec(compile(oldsource,'original-wheel/pack.py','exec'),mod.__dict__)
encoder=ONNXPooledEncoder(manifest['embedding']['path'])
assert encoder.fingerprint==manifest['embedding']['fingerprint']
representatives={r['case_id']:r for r in read_rows(parent/'rows.jsonl') if r['profile']=='nsn_lexical'}
results=[]
for case,row in representatives.items():
 info=row['corpus_cache'];cache=Path(manifest['corpus_cache_path'])/info['key']
 for name,sha in info['file_sha256'].items():assert hashlib.sha256((cache/name).read_bytes()).hexdigest()==sha
 with tempfile.TemporaryDirectory() as tmp:
  with Runtime(tmp) as rt:
   with closing(sqlite3.connect('file:'+ (cache/'events.sqlite').as_posix()+'?mode=ro',uri=True)) as src:
    with rt.storage._runtime_connection() as dst:src.backup(dst)
   with closing(sqlite3.connect(rt.db_path)) as c:
    native=c.execute('SELECT id,content FROM events ORDER BY rowid').fetchall()
   # Cache vector order is original insertion order, verified by native row count.
   vectors=np.load(cache/'vectors.npy',allow_pickle=False)
   assert len(native)==info['native_events']==len(vectors)
   lookup={id:{'id':id,'content':content} for id,content in native}
   vs=CompactVectorStore.from_encoder(rt.db_path,encoder)
   vs.add_batch([(id,vec.tolist(),'development','event') for (id,_),vec in zip(native,vectors)])
   for semantic in (False,True):
    for version,cls in (('before',mod.ContextPacker),('after',ContextPacker)):
     packer=cls(rt.storage,vector_store=vs if semantic else None,encoder=encoder if semantic else None)
     pack=packer.retrieve_pack(row['question'],namespace='development',max_tokens=512,use_graph=False,use_summaries=False)
     candidates=[lookup[i.id] for i in pack.items if i.id in lookup]
     # Content already includes speaker/date. Match the original 512-byte repack.
     used=0;ids=[]
     for item in candidates:
      size=len((item['content']+'\n').encode('utf-8'))
      if used+size<=512:ids.append(item['id']);used+=size
     gold=set(row['gold_ids'])
     results.append({'case_id':case,'dataset':row['dataset'],'mode':'semantic' if semantic else 'minimal','version':version,'gold_ids':sorted(gold),'selected_ids':ids,'evidence_recall':len(gold&set(ids))/len(gold) if gold else None,'complete_support':gold<=set(ids) if gold else None,'pack_tokens':pack.token_count,'budget':pack.token_budget,'cache_key':info['key']})
summary={}
for dataset in ('locomo','longmemeval'):
 for mode in ('minimal','semantic'):
  for version in ('before','after'):
   part=[r for r in results if r['dataset']==dataset and r['mode']==mode and r['version']==version and r['evidence_recall'] is not None]
   summary[f'{dataset}:{mode}:{version}']={'questions_with_evidence':len(part),'recall':statistics.mean(r['evidence_recall'] for r in part),'complete_support':statistics.mean(r['complete_support'] for r in part)}
record={'scope':'Exploratory memory-only diagnostic on the previous 15-question sample; no new answer accuracy or production-quality claim','native_cache':'Inherited checksum-verified original native database and neural embeddings, private copies; no generated memories or old answers used as inputs','old_wheel_sha256':hashlib.sha256(oldwheel.read_bytes()).hexdigest(),'new_wheel_sha256':hashlib.sha256(Path('dist/prototype-hardening-20261004/nsn-0.3.0-py3-none-any.whl').read_bytes()).hexdigest(),'summary':summary,'rows':results}
(root/'memory-probe.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
