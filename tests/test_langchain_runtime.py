import asyncio
import pytest
from nsn import Runtime

pytest.importorskip('langchain_core')
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from neurosleepnet.integrations.langchain import NeurosleepNetHistory, wrap_runnable

def test_real_history_persistence_clear_and_runnable_lifecycle(tmp_path):
    rt = Runtime(str(tmp_path))
    history = NeurosleepNetHistory(runtime=rt,namespace='a',session_id='one')
    history.add_messages([HumanMessage(content='gateway port 8080'),AIMessage(content='acknowledged')])
    asyncio.run(history.aadd_messages([HumanMessage(content='gateway host internal')]))
    restarted = NeurosleepNetHistory(runtime=rt,namespace='a',session_id='one')
    assert [m.content for m in restarted.messages]==['gateway port 8080','acknowledged','gateway host internal']
    different = NeurosleepNetHistory(runtime=rt,namespace='a',session_id='two')
    different.add_messages([HumanMessage(content='retain session two')])
    assert NeurosleepNetHistory(runtime=rt,namespace='b').messages==[]
    model = wrap_runnable(RunnableLambda(lambda text:text),rt,namespace='a')
    assert '<retrieved_evidence>' in model.invoke('gateway port')
    assert '8080' in asyncio.run(model.ainvoke('gateway port'))
    asyncio.run(history.aclear())
    assert restarted.messages==[]
    assert different.messages[0].content=='retain session two'
