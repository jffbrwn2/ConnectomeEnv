"""Verified, provider-neutral visual demonstrations supplied before an episode."""
import hashlib
import json
from pathlib import Path


def load_context(manifest, provider):
    from .harness import content
    path = Path(manifest).resolve()
    spec = json.loads(path.read_text())
    if spec.get('schema') != 'visual-demonstrations-v1' or not spec.get('examples'):
        raise ValueError('Invalid visual demonstration manifest')
    blocks = []
    text_type = 'input_text' if provider == 'responses' else 'text'
    blocks.append({'type':text_type, 'text':spec['introduction']})
    seen = set()
    for example in spec['examples']:
        if example['case_id'] in seen:
            raise ValueError('Duplicate demonstration case')
        seen.add(example['case_id'])
        image = (path.parent / example['image']).resolve()
        if not image.is_relative_to(path.parent / 'frames') or image.suffix.lower() != '.png':
            raise ValueError('Demonstration image must be a PNG inside frames')
        if hashlib.sha256(image.read_bytes()).hexdigest() != example['sha256']:
            raise ValueError('Demonstration image changed')
        blocks.extend(content({'example':len(seen), 'description':example['caption'],
                               'image':example['image']}, path.parent, provider))
    blocks.append({'type':text_type,'text':spec.get('closing', 'End of examples. The next observation is the independent task to solve. Do not assume its anatomy or required edits match an example.')})
    return [{'role':'user','content':blocks}]
