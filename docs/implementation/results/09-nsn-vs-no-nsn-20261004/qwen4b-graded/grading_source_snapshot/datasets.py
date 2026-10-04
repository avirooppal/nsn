import hashlib
import json
import re
from pathlib import Path
from collections import Counter

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def load_native(path, dataset):
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    return convert_native(data,dataset)


def iter_native(path,dataset):
    """Convert one native conversation at a time for multi-GB haystacks."""
    import ijson
    with Path(path).open('rb') as handle:
        for record in ijson.items(handle,'item'):
            yield from convert_native([record],dataset)


def convert_native(data,dataset):
    cases=[]
    for record in data:
        events=[]
        if dataset=='locomo':
            group='locomo:'+str(record['sample_id'])
            conversation=record['conversation']
            sessions=sorted((k for k in conversation if re.fullmatch(r'session_\d+',k)),key=lambda s:int(s.split('_')[1]))
            for session in sessions:
                date=conversation[session+'_date_time']
                for turn in conversation[session]:
                    events.append({'id':turn['dia_id'],'session':session,'date':date,'role':'user','speaker':turn['speaker'],'content':turn['text'],'caption':turn.get('blip_caption'),'native':turn})
            for index,qa in enumerate(record['qa']):
                cases.append({'id':group+f':{index:04d}','group':group,'dataset':dataset,'category':str(qa['category']),'question':qa['question'],'answer':qa.get('answer'),'question_date':None,'events':events,'gold_ids':list(qa.get('evidence',[])),'evidence_unit':'dialog','native_qa':qa})
        elif dataset=='longmemeval':
            if not len(record['haystack_sessions'])==len(record['haystack_dates'])==len(record['haystack_session_ids']):
                raise ValueError('Unaligned native sessions/dates')
            counts=Counter(record['haystack_session_ids']);occurrences=Counter()
            for session,date,turns in zip(record['haystack_session_ids'],record['haystack_dates'],record['haystack_sessions']):
                occurrence=occurrences[session];occurrences[session]+=1
                identity=session if counts[session]==1 else f'{session}#occurrence={occurrence}'
                for index,turn in enumerate(turns):
                    events.append({'id':identity+f':{index}','session':session,'date':date,'role':turn['role'],'speaker':turn['role'],'content':turn['content'],'native':turn})
            gold=[e['id'] for e in events if e['native'].get('has_answer')]
            cases.append({'id':'longmemeval:'+record['question_id'],'group':'longmemeval:'+record['question_id'],'dataset':dataset,'category':record['question_type'],'question':record['question'],'answer':record['answer'],'question_date':record['question_date'],'events':events,'gold_ids':gold,'evidence_unit':'has_answer_turn' if gold else 'unannotated','answer_session_ids':record['answer_session_ids'],'native_qa':{k:v for k,v in record.items() if k!='haystack_sessions'}})
        else:
            raise ValueError('Unknown native dataset')
    if len({c['id'] for c in cases})!=len(cases):
        raise ValueError('Duplicate question IDs')
    return cases

def freeze(cases, seed=42):
    """Union conversations sharing session IDs or exact session content before split."""
    parent={c['group']:c['group'] for c in cases}
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]
            x=parent[x]
        return x
    seen={}
    for c in cases:
        sessions={}
        for e in c['events']:
            sessions.setdefault(e['session'],[]).append((e['date'],e['role'],e['speaker'],e['content']))
        for sid,turns in sessions.items():
            for key in (c['dataset']+':id:'+sid if c['dataset']=='longmemeval' else c['group']+':'+sid,c['dataset']+':content:'+digest(turns)):
                if key in seen:
                    a,b=find(c['group']),find(seen[key])
                    parent[max(a,b)]=min(a,b)
                seen[key]=c['group']
    clusters={g:find(g) for g in parent}
    groups=sorted(set(clusters.values()),key=lambda g:digest([seed,g]))
    dev=set(groups[:max(1,len(groups)//5)])
    # Freeze all questions, never select on observed answers or retrieval success.
    split={c['id']:('development' if clusters[c['group']] in dev else 'evaluation') for c in cases}
    event_hashes={}
    compact=[]
    for c in cases:
        event_hashes.setdefault(id(c['events']),None)
        if event_hashes[id(c['events'])] is None:
            event_hashes[id(c['events'])]=digest(c['events'])
        compact.append({**{k:v for k,v in c.items() if k not in ('events','native_qa')},'events_sha256':event_hashes[id(c['events'])]})
    return {'seed':seed,'tuning_trials':0,'cases_sha256':digest(compact),'clusters':clusters,'split':split,'evaluation_questions':sum(v=='evaluation' for v in split.values()),'development_questions':sum(v=='development' for v in split.values())}

def smoke_cases(cases, frozen, per_dataset=2):
    selected=[]
    for dataset in sorted({c['dataset'] for c in cases}):
        groups=set()
        for c in sorted(cases,key=lambda c:c['id']):
            cluster=frozen['clusters'][c['group']]
            if c['dataset']==dataset and frozen['split'][c['id']]=='development' and cluster not in groups:
                selected.append(c);groups.add(cluster)
                if len(groups)>=per_dataset: break
    return selected
