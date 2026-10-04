# Retired nonessential repository material — 2026-10-04

Removed 15 obsolete top-level files/folders (38 tracked files; 3,627,488 bytes) from the working repository after checking package discovery, imports, tests, CI, Docker and active documentation references:

- frontend/ and static/: standalone dashboards; no SDK/test dependency.
- server.py: old cloud studio using the deprecated initialization path. The supported authenticated REST factory remains in neurosleepnet/integrations/api.py.
- paper_figures/, generate_paper_figures.py, neurosleepnet_ieee_paper.tex, paper_overleaf.tex and nsn_overleaf_upload.zip: standalone publication artifacts, not runtime/build requirements.
- hardcore_benchmark.py, ollama_poc_benchmark.py, ollama_vanilla_nsn_demo.py and the three root benchmark PNG files: superseded experiment/demo assets. Current benchmarks/, examples/ and measured reports remain.
- requirements.txt: obsolete mandatory-ML dependency list. pyproject.toml remains the dependency authority, with optional extras.

The recovery copy is outside the repository at `C:/Users/aviroop/AppData/Local/Temp/nsn-retired-artifacts-20261004`. retired-files.json records every original path, size, SHA-256 and archive path. All 38 archive files were independently checksum-verified and the originals are absent from the working tree. Their Git history remains intact. This reversible removal avoids the recursive-deletion policy rejection encountered earlier. Restore selected items by copying their archived paths back to the original locations after checking those locations are absent.

Kept: nsn/, neurosleepnet/, supported optional integrations, tests/, benchmarks/, examples/, scripts/, .github/, Docker files, pyproject.toml, uv.lock, plan.md, documentation, original raw evaluation evidence, verified wheel candidates and live memory databases. Active nsn.egg-info metadata also remains. Git diff verifies supported source/tests/build configuration is unchanged relative to HEAD.

README now describes the removal and current supported REST/dependency paths. plan.md and the historical README label obsolete paths as historical inspection material. No SDK behavior or tests were modified.

## Validation

- 300 passed, 2 skipped, 1 deselected, 3 warnings in 102.28s (0:01:42).
- Clean temporary wheel build succeeds; all nsn/ and neurosleepnet/ payload bytes match the previously verified SDK wheel, and updated README is included in metadata. See retired-build.txt and retired-build-audit.json.
- Exact clean wheel installed into a fresh minimal Python 3.12 environment outside the checkout. Network-denied imports/operation, restart recall, namespace isolation, no ML dependencies and package configuration pass. See retired-installed.txt.
- Docker build from the actual cleaned repository succeeds; its ephemeral restart fixture passes with `--network none`. See retired-docker-build.txt and retired-docker-smoke.txt. Existing memory volumes were not used.
- Original verified wheel/checksums and earlier evaluation/performance artifacts remain preserved. Performance exclusion does not qualify hardware gates; production/evaluation/license gaps remain unchanged.

Commands:
```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -B -m pytest tests/ benchmarks/tests/ -m 'not performance' -q -p no:cacheprovider
docker build -t nsn-cleanup-check:20261004 .
docker run --rm --network none nsn-cleanup-check:20261004
python -B -m build --wheel --outdir C:/Users/aviroop/AppData/Local/Temp/nsn-retired-build-check-20261004/dist C:/Users/aviroop/AppData/Local/Temp/nsn-retired-build-check-20261004/source
```

Temporary build source consists of pyproject.toml, README.md, nsn/ and neurosleepnet/ copied from the cleaned checkout. Installed validation runs scripts/verify_install.py using the temporary venv from outside the repository. Source tests used PYTHONDONTWRITEBYTECODE=1 to avoid recreating caches.

Captured `retired-*` artifacts retain exact bytes through the directory's Git attributes. In retired-checksums.json, the three source-document hashes refer to Git-normalized stored content; reproduce those from `git show HEAD:<path>` rather than platform-converted working copies. Artifact hashes cover the captured files directly.
