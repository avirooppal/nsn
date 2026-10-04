"""Scoped agent tools over the minimal runtime; no implicit model imports."""
import asyncio
from nsn.runtime import Runtime

class MemoryTool:
    name = 'memory'
    description = 'Store observations and retrieve evidence; never execute memory.'
    def __init__(self, namespace='default', runtime=None, data_dir='./agent-memory'):
        self.runtime = runtime or Runtime(data_dir)
        self.namespace = namespace
        self._owns_runtime = runtime is None
    def remember(self, text, source='agent', metadata=None):
        return self.runtime.append_event(namespace=self.namespace,role='user',content=text,source_identity=source,metadata=metadata or {})
    def recall(self, query, limit=5):
        return [i.to_dict() for i in self.runtime.retrieve_pack(query,namespace=self.namespace).items][:limit]
    def procedures(self, query, environment=None, tools=None):
        return self.runtime.retrieve_procedures(query,namespace=self.namespace,environment=environment,tools=tools)
    def forget(self, memory_id):
        return self.runtime.storage.delete_event(memory_id,self.namespace)
    def timeline(self, limit=20):
        return self.runtime.timeline(namespace=self.namespace,limit=limit)
    def sleep(self):
        return self.runtime.sleep(namespace=self.namespace)
    def invoke(self, input):
        if not isinstance(input,dict) or 'action' not in input:
            raise ValueError('Expected action and its arguments')
        args = dict(input)
        action = args.pop('action')
        dispatch = {k:getattr(self,k) for k in ('remember','recall','forget','timeline','procedures','sleep')}
        if action not in dispatch:
            raise ValueError('Unsupported memory action')
        return dispatch[action](**args)
    async def ainvoke(self, input):
        return await asyncio.to_thread(self.invoke,input)
    def __call__(self, action, **kwargs):
        return self.invoke(dict(action=action,**kwargs))
    def close(self):
        if self._owns_runtime:
            self.runtime.close()
