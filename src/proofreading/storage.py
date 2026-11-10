"""Explicit mount relocation; never changes frozen hashes or scoring contents."""
import json
import os
from pathlib import Path


def path_map():
    mapping = json.loads(os.environ.get('CONNECTOME_PATH_MAP', '{}'))
    if not isinstance(mapping, dict) or any(not isinstance(k, str) or not isinstance(v, str)
            or not Path(k).is_absolute() or not Path(v).is_absolute() for k, v in mapping.items()):
        raise ValueError('CONNECTOME_PATH_MAP must map absolute directory prefixes to absolute mounts')
    return mapping


def data_path(value, base=None):
    path = Path(value)
    if not path.is_absolute():
        path = Path(base or '.') / path
    path = path.resolve()
    for source, target in sorted(path_map().items(), key=lambda item: len(Path(item[0]).parts), reverse=True):
        if path.is_relative_to(source):
            return Path(target) / path.relative_to(source)
    return path
