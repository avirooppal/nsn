import json
from pathlib import Path
import subprocess
import sys


def test_resource_monitor_drains_outputs_larger_than_pipe_capacity(tmp_path):
    destination=tmp_path/'profile.json'
    code="from benchmarks import release_profile as p; import sys; p.CHILD=\"import json,sys;sys.stderr.write('e'*200000);print(json.dumps({'padding':'x'*200000}))\";sys.argv=['profile','--python',sys.executable,'--output',sys.argv[1],'--records','1'];p.main()"
    result=subprocess.run([sys.executable,'-c',code,str(destination)],cwd=Path(__file__).resolve().parents[2],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert len(json.loads(destination.read_text(encoding='utf-8'))['padding'])==200000
