"""Bounded-memory full-corpus freezing; no oracle split reused for distractors."""
from benchmarks.public_eval.datasets import digest, iter_native
from pathlib import Path

VARIANTS={'oracle':'longmemeval_oracle.json','s':'longmemeval_s_cleaned.json','m':'longmemeval_m_cleaned.json'}

def factory(root,variant):
    if variant not in VARIANTS:raise ValueError('Unknown LongMemEval variant')
    root=Path(root)
    def cases():
        yield from iter_native(root/'locomo__data__locomo10.json','locomo')
        yield from iter_native(root/('longmemeval_data__'+VARIANTS[variant]),'longmemeval')
    return cases

def freeze_stream(cases,seed=42):
    parent={};seen={};compact=[];index=[];group_hashes={};ids=set()
    def find(group):
        while parent[group]!=group:
            parent[group]=parent[parent[group]];group=parent[group]
        return group
    for case in cases:
        if case['id'] in ids:raise ValueError('Duplicate question ID')
        ids.add(case['id']);group=case['group'];parent.setdefault(group,group)
        if group not in group_hashes:
            group_hashes[group]=digest(case['events']);sessions={}
            for event in case['events']:
                sessions.setdefault(event['session'],[]).append((event['date'],event['role'],event['speaker'],event['content']))
            for session,turns in sessions.items():
                for key in (case['dataset']+':id:'+session if case['dataset']=='longmemeval' else group+':'+session,case['dataset']+':content:'+digest(turns)):
                    if key in seen:
                        a,b=find(group),find(seen[key]);parent[max(a,b)]=min(a,b)
                    seen[key]=group
        index.append({k:case[k] for k in ('id','group','dataset')})
        compact.append({**{k:v for k,v in case.items() if k not in ('events','native_qa')},'events_sha256':group_hashes[group]})
    clusters={g:find(g) for g in parent}
    groups=sorted(set(clusters.values()),key=lambda g:digest([seed,g]))
    dev=set(groups[:max(1,len(groups)//5)])
    split={c['id']:('development' if clusters[c['group']] in dev else 'evaluation') for c in index}
    return {'seed':seed,'tuning_trials':0,'cases_sha256':digest(compact),'clusters':clusters,'split':split,'evaluation_questions':sum(v=='evaluation' for v in split.values()),'development_questions':sum(v=='development' for v in split.values())},index
