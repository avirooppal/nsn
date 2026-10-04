"""Optional LangChain history and text Runnable adapter over shared runtime."""
import asyncio
from nsn.runtime import Runtime
try:
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
except ImportError:
    BaseChatMessageHistory = object

class NeurosleepNetHistory(BaseChatMessageHistory):
    def __init__(self, namespace='default',session_id='default',runtime=None,data_dir='./agent-memory'):
        if BaseChatMessageHistory is object:
            raise ImportError('Install nsn[langchain] for message history')
        self.runtime = runtime or Runtime(data_dir)
        self.namespace = namespace
        self.session_id = session_id
        self._owns_runtime = runtime is None
    @property
    def messages(self):
        rows = self.runtime.timeline(namespace=self.namespace,session_id=self.session_id,limit=10000)
        classes = {'user':HumanMessage,'assistant':AIMessage,'system':SystemMessage}
        return [classes[r['role']](content=r['content']) for r in rows if r['role'] in classes and r['turn_status']=='completed']
    def add_messages(self, messages):
        roles = {'human':'user','ai':'assistant','system':'system'}
        for m in messages:
            if m.type not in roles or not isinstance(m.content,str):
                raise TypeError('History supports text human, ai and system messages')
            self.runtime.append_event(namespace=self.namespace,session_id=self.session_id,role=roles[m.type],content=m.content,source_identity='langchain')
    def clear(self):
        self.runtime.advanced.delete_session(self.session_id,self.namespace)
    async def aget_messages(self):
        return await asyncio.to_thread(lambda:self.messages)
    async def aadd_messages(self, messages):
        await asyncio.to_thread(self.add_messages,messages)
    async def aclear(self):
        await asyncio.to_thread(self.clear)
    def close(self):
        if self._owns_runtime:
            self.runtime.close()

def wrap_runnable(runnable,runtime,namespace='default',**kwargs):
    """Text-input Runnables use nsn.wrap's prior-recall and failure lifecycle."""
    return runtime.wrap(runnable,namespace=namespace,**kwargs)
