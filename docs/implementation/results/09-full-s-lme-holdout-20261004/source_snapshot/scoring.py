"""Execute only pinned upstream scoring functions, without heavy judge imports."""
import ast
from collections import Counter
import hashlib
from pathlib import Path
import string

def functions(path,names,expected_sha256, namespace=None):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_sha256:
        raise ValueError('Official scoring source checksum mismatch')
    tree=ast.parse(raw)
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in selected}!=set(names):
        raise ValueError('Missing official scoring functions')
    scope=dict(namespace or {})
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),scope)
    return scope

def official_locomo(path,sha256):
    import regex
    import numpy
    from nltk.stem import PorterStemmer
    return functions(path,('normalize_answer','f1_score','f1','eval_question_answering'),sha256,{'regex':regex,'np':numpy,'ps':PorterStemmer(),'Counter':Counter,'string':string})['eval_question_answering']

def official_longmem_prompt(path,sha256):
    return functions(path,('get_anscheck_prompt',),sha256)['get_anscheck_prompt']
