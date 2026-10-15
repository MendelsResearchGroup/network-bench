"""Run dataset analysis explicitly, from the repository root."""
import argparse
import json
from pathlib import Path
from . import measure_difficulty

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('dataset')
parser.add_argument('--root', type=Path, default=Path('results'))
parser.add_argument('--systems', type=int, default=20)
parser.add_argument('--manifest', default='clean')
parser.add_argument('--history', type=int, default=3)
parser.add_argument('--count', type=int, default=15)
parser.add_argument('--max-frames', type=int)
args = parser.parse_args()
result = measure_difficulty(args.dataset, systems=args.systems, manifest=args.manifest or None,
                           history=args.history, count=args.count, max_frames=args.max_frames)
path = args.root / args.dataset / 'difficulty.json'
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(result, indent=1))
print(path)
