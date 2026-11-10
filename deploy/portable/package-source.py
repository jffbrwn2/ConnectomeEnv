"""Build a small release source archive from an explicit allowlist (no data/secrets)."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
root=Path(__file__).resolve().parents[2]
a.out.mkdir(parents=True,exist_ok=True)
files=[root/n for n in ['pyproject.toml','README.md','LICENSE','pytest.ini',
    'docs/RELEASE_QUICKSTART.md','docs/PORTABLE_EVALUATION.md','docs/PORTABLE_RC2_VALIDATION.md']]
for pattern in ['src/proofreading/*.py','src/proofreading/*.html','tests/proofreading/*.py',
                'deploy/portable/*','configs/proofreading/*.example.json']:
    files.extend(root.glob(pattern))
for name in ['prepare_balanced_cb2','prepare_cb2_practice','prepare_indexed_world',
             'prepare_matched_cb2','run_cb2_practice','watch_comparison_scores','comparison_batch']:
    files.append(root/'scripts/proofreading'/f'{name}.py')
files=sorted(set(f for f in files if f.is_file()))
archive=a.out/'connectomeenv-0.1.0rc2-source.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for file in files:tar.add(file,arcname=str(file.relative_to(root)),recursive=False)
(a.out/'source-files.json').write_text(json.dumps({str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2))
print(archive)
