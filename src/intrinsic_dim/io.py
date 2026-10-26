"""Atomic metadata writes and portable provenance."""
import csv
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import numpy as np

def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k,v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)): return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, Path): return str(value)
    return value

def write_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(clean(data), indent=2, allow_nan=False) + '\n')
    tmp.replace(path)

def read_json(path): return json.loads(Path(path).read_text())
def fingerprint(value): return hashlib.sha256(json.dumps(clean(value), sort_keys=True).encode()).hexdigest()
def versions():
    result = {'python': platform.python_version(), 'platform': platform.platform()}
    for name in ['numpy','scipy','matplotlib','PyYAML','torch','transformers','tokenizers','huggingface-hub']:
        try: result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: result[name] = None
    return result

def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for row in rows: writer.writerow({k: json.dumps(clean(v)) if isinstance(v,(dict,list)) else clean(v) for k,v in row.items()})


def source_fingerprint():
    """Digest source files for reproducibility even outside a Git checkout."""
    root=Path(__file__).parent
    return fingerprint({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.py'))})

def dataset_input_fingerprint(rows):
    """Freeze authored inputs while allowing tokenizer-derived metadata to change."""
    return fingerprint([{k:v for k,v in row.items() if k not in ['token_count','token_count_method']} for row in rows])
