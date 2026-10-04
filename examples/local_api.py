"""Set NSN_API_TOKEN to a random credential, then run this localhost example."""
import os
from neurosleepnet.integrations.api import create_app

if __name__ == '__main__':
    import uvicorn
    app = create_app(tokens={os.environ['NSN_API_TOKEN']:'local-agent'})
    uvicorn.run(app,host='127.0.0.1',port=8000)
