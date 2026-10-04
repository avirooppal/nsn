"""Regenerate verification summary from raw logs, installed checks and wheel bytes."""
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
WHEEL = REPO/'dist/memory-verification-20261004/nsn-0.3.0-py3-none-any.whl'
PREVIOUS = REPO/'dist/prototype-hardening-20261004/nsn-0.3.0-py3-none-any.whl'
sha = hashlib.sha256(WHEEL.read_bytes()).hexdigest()
with zipfile.ZipFile(WHEEL) as archive, zipfile.ZipFile(PREVIOUS) as previous:
    sdk = [n for n in archive.namelist() if n.startswith(('nsn/','neurosleepnet/')) and n.endswith('.py')]
    assert all(archive.read(n)==(REPO/n).read_bytes() for n in sdk)
    changed = [n for n in sdk if archive.read(n)!=previous.read(n)]
    metadata = archive.read('nsn-0.3.0.dist-info/METADATA').decode()
    assert not re.search(r'^License(?:-Expression)?:',metadata,re.M)
functional = (ROOT/'functional-final.txt').read_text(encoding='utf-8-sig')
summary = re.search(r'300 passed, 2 skipped, 1 deselected, 3 warnings in [^\n]+',functional).group().strip()
windows = json.loads((ROOT/'windows-matrix.json').read_text())
confirmation = json.loads((ROOT/'windows-matrix-confirmation.json').read_text())
linux = json.loads((ROOT/'linux-matrix.json').read_text())
for matrix in (windows,confirmation,linux):
    assert all(v['wheel_sha256']==sha for v in matrix.values())
onnx = json.loads((ROOT/'onnx-offline.json').read_text())
local = json.loads((ROOT/'local-model-smoke.json').read_text())
minimal = json.loads((ROOT/'minimal-installed.txt').read_text(encoding='utf-8-sig').strip())
assert onnx['passed'] and minimal['restart_recall'] and local['restart_injection_passed']
assert '23 passed' in (ROOT/'onnx-installed-tests.txt').read_text(encoding='utf-8-sig')
result = {'wheel':str(WHEEL.relative_to(REPO)), 'wheel_sha256':sha,'wheel_bytes':WHEEL.stat().st_size,
    'sdk_source_matches_wheel':True,'changed_sdk_files_since_previous_candidate':changed,
    'functional':summary,'new_memory_contract_tests':18,'fresh_onnx_tests':'23 passed',
    'windows_first_attempt':{k:v['passed'] for k,v in windows.items()},
    'windows_confirmation':{k:v['passed'] for k,v in confirmation.items()},
    'linux':{k:v['passed'] for k,v in linux.items()},
    'minimal_offline':minimal,'semantic_offline':onnx,
    'local_model_smoke':{k:v for k,v in local.items() if k!='calls'},
    'unresolved':['First concurrent Windows 3.9 optional enrichment failure retained; root cause not proven',
       'Current candidate performance not requalified; earlier failed resource gates remain open',
       'Full public evaluations, competitors, human review, hosted CI and owner license remain incomplete']}
(ROOT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
lines = ['# Memory-layer verification — 2026-10-04','',
    'Core memory behavior is verified on the installed candidate. This is functional prototype evidence, not complete production qualification or model-intelligence/competitor claims.','',
    f'Wheel: `{result["wheel"]}`, {result["wheel_bytes"]:,} bytes, SHA-256 `{sha}`. SDK source bytes match the wheel. Owner license remains unresolved; nothing published.','',
    '## Repairs and meaningful tests','',
    'Added 18 memory-contract tests. They check prior-memory injection with one model call, no self-recall, separate-process persistence, namespaces, history larger than context, token bounds, fact corrections and provenance deletion, export/restore/delete scope, encoder failure and durable replay, lifecycle guards, dense filtering scores/cache behavior, and prepared real ONNX offline scoping. Deterministic hash encoders and spy models verify mechanics; they are not semantic-quality or answer-accuracy evidence.','',
    'Reproduced and repaired future raw events appearing in historical retrieval, wrong-session dense hits exhausting top-k, and indexing/rebuild/relation/pin writes after runtime close. Temporal raw-event filtering uses event_time (otherwise observed_at) and UTC-aware SQLite date comparisons before limits. Dense event eligibility is applied before top-k while preserving matrix cache and score mapping. Namespace facts remain shared across sessions; session filters constrain events and are not tenant security boundaries. Scoped dense filtering scans eligible IDs, so its large-scale latency is not qualified.','',
    '## Actual verification results','',f'- Full functional suite: **{summary}**. One hardware test deliberately deselected; two skips retained.',
    '- Fresh minimal wheel outside checkout: network denied, no ML dependencies, unchanged import threads, restart recall, namespace isolation and bundled config passed.',
    '- Fresh `[semantic-onnx]` installation outside checkout: **23 passed** (18 memory contracts and 5 encoder tests); prepared-model offline durable replay passed without Torch/Transformers imports.',
    f'- Windows first attempt: `{result["windows_first_attempt"]}`; separate confirmation: `{result["windows_confirmation"]}`. Linux containers: `{result["linux"]}`. Core cells use installed wheels with explicit REST/legacy exclusions and unavailable ONNX asset skips.',
    '- First Windows 3.9 run had 99 passed, 2 skipped, 2 deselected and one optional enrichment failure during concurrent workloads. The failure remains in windows-matrix.json. No assertion, workload, timeout or resource limit was relaxed. Its root cause is not proven; separate confirmation is recorded independently.',
    '- Actual local `qwen3:1.7b`: stored nonce recalled on both subsequent and post-restart turns, prior evidence injected and other namespace empty. Single synthetic smoke only, not a comparable accuracy benchmark. See local-model-smoke.json for actual prompts/responses/model digest.','',
    '## Reproduce','', 'Run from the repository unless stated otherwise:', '```powershell',
    "python -m pytest tests/ benchmarks/tests/ -m 'not performance' -q -p no:cacheprovider",
    'python scripts/local_matrix.py --wheel dist/memory-verification-20261004/nsn-0.3.0-py3-none-any.whl --output windows-matrix-confirmation.json',
    'python scripts/local_linux_matrix.py --wheel dist/memory-verification-20261004/nsn-0.3.0-py3-none-any.whl --output linux-matrix.json',
    'python docs/implementation/results/10-memory-verification-20261004/regenerate.py', '```','',
    'Install this wheel into new environments with no inherited packages. From outside the checkout run scripts/verify_install.py in the minimal environment, scripts/verify_onnx.py with prepared assets in the semantic environment, and absolute-path tests/test_memory_layer_contract.py plus tests/test_onnx_encoder.py there. local-model-smoke.py requires the existing local Ollama server and pinned model; no downloads are performed.','',
    '## Remaining gates','',
    'Phases 09 and 10 remain PARTIAL. Historical performance misses are retained in ../10-prototype-hardening-20261004/readiness.json; they are not measurements of this new candidate. Fresh-query latency, default inline semantic resource qualification, full public/competitor evaluation, human grading, hosted CI and source-license choice remain open. No hardware assertion was weakened or marked passed by functional verification.']
(ROOT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
checksums = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.iterdir()) if p.is_file() and p.name!='checksums.json'}
checksums[str(WHEEL.relative_to(REPO))]=sha
(ROOT/'checksums.json').write_text(json.dumps(checksums,indent=2),encoding='utf-8')
print(json.dumps({k:result[k] for k in ('wheel_sha256','wheel_bytes','functional','windows_confirmation','linux','changed_sdk_files_since_previous_candidate')},indent=2))
