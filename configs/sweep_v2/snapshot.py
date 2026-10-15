"""Freeze prepared configs and Python source for PBS jobs."""
import hashlib
import json
import shutil


def create_snapshot(repo):
    files = sorted(path for folder in ('src', 'configs') for path in (repo / folder).rglob('*')
                   if path.is_file() and '__pycache__' not in path.parts
                   and path.suffix in ('.py', '.json', '.pbs', '.md', '.typed'))
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(repo)).encode())
        digest.update(path.read_bytes())
    revision = digest.hexdigest()
    root = repo / 'results/sweep-v2'
    target = root / 'snapshots' / revision
    if not target.exists():
        staging = target.with_name(revision + '.tmp')
        for path in files:
            destination = staging / path.relative_to(repo)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        staging.rename(target)
    marker = root / 'snapshot.json'
    temporary = marker.with_suffix('.tmp')
    temporary.write_text(json.dumps({'path': str(target), 'sha256': revision}, indent=2) + '\n')
    temporary.replace(marker)
    return target
