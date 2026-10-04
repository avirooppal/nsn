"""Optional authenticated REST factory; importing never creates runtime/server."""
from contextlib import asynccontextmanager
import hmac

def create_app(runtime=None,data_dir='./agent-memory',tokens=None):
    """Host-owned token-to-namespace mapping. Bind localhost; remote use needs TLS.

    Backups, outcome confirmation, deletion and enrichment remain host operations.
    """
    try:
        from fastapi import FastAPI, HTTPException, Request
    except ImportError as exc:
        raise ImportError('Install nsn[api]') from exc
    from nsn.runtime import Runtime
    if not isinstance(tokens,dict) or not tokens or not all(isinstance(k,str) and len(k)>=16 and isinstance(v,str) and v for k,v in tokens.items()):
        raise ValueError('Provide tokens of at least 16 characters mapped to namespaces')
    credentials = dict(tokens)
    owns_runtime = runtime is None
    rt = runtime or Runtime(data_dir)
    @asynccontextmanager
    async def lifespan(app):
        yield
        if owns_runtime:
            rt.close()
    app = FastAPI(lifespan=lifespan)
    def scope(request):
        supplied = request.headers.get('authorization','').removeprefix('Bearer ')
        for token, namespace in credentials.items():
            if hmac.compare_digest(supplied.encode('utf-8'),token.encode('utf-8')):
                return namespace
        raise HTTPException(401,'Invalid bearer token')
    @app.get('/health')
    def health():
        return {'status':'ok'}
    @app.post('/observe')
    async def observe(request: Request):
        ns = scope(request)
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw)>65536:
                raise HTTPException(413,'Observation exceeds 64 KiB')
        import json
        try:
            data = json.loads(raw)
            if not isinstance(data,dict) or set(data)!={'content'} or not isinstance(data['content'],str) or not data['content'].strip():
                raise ValueError()
        except (ValueError,TypeError):
            raise HTTPException(422,'Expected content only; namespace bound to credentials')
        return {'id':rt.append_event(namespace=ns,role='user',content=data['content'],source_identity='rest')}
    @app.get('/search')
    def search(request: Request,q: str):
        ns = scope(request)
        if len(q)>4096 or set(request.query_params)-{'q'}:
            raise HTTPException(422,'Invalid query or scope override')
        return rt.explain(q,namespace=ns)
    @app.get('/timeline')
    def timeline(request: Request):
        ns = scope(request)
        if request.query_params:
            raise HTTPException(422,'Scope fixed by credentials')
        return rt.timeline(namespace=ns)
    return app
