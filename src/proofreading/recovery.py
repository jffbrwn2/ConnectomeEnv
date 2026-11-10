"""Durable provider exchanges and verified recovery into a fresh local episode.

Recovery replays saved model outputs without paid requests. Local tools rebuild
an isolated session; exact request hashes verify the images and history before
any new inference. No historical images or messages are removed.
"""
import errno
import hashlib
import json
import os
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError


def atomic_json(path, value):
    path = Path(path)
    value = {**value, "updated_at_unix": time.time()}
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as f:
        os.chmod(temporary, 0o600)
        json.dump(value, f)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def failure_state(exc):
    detail = getattr(exc, "provider_detail", {})
    if isinstance(exc, OSError) and exc.errno in (errno.ENOSPC, errno.EDQUOT):
        reason = "storage_exhausted"
    elif detail.get("billing_action_required"):
        reason = "billing"
    elif detail.get("reason") == "image_count_limit":
        reason = "image_count_limit"
    elif isinstance(exc, HTTPError) and exc.code in (401, 403):
        reason = "authentication"
    elif detail.get("retryable") or (isinstance(exc, HTTPError) and exc.code in (500, 502, 503, 504, 520, 529)) or isinstance(exc, (TimeoutError, ConnectionError)) or (isinstance(exc, URLError) and not isinstance(exc, HTTPError)):
        reason = "transient_provider_error"
    elif isinstance(exc, HTTPError):
        reason = "provider_request_rejected"
    else:
        reason = "runtime_error"
    return {"status": "paused", "reason": reason,
            "retryable": reason == "transient_provider_error",
            "error_type": type(exc).__name__, "provider_detail": detail}


class JournalTransport:
    def __init__(self, directory, transport, resume_from=None):
        self.directory = Path(directory) / "exchanges"
        self.directory.mkdir(exist_ok=True)
        self.transport = transport
        self.source = Path(resume_from) / "exchanges" if resume_from else None
        self.index = 0
        self.replayed = 0
        self.last_replayed = False
        if self.source is not None and not self.source.is_dir():
            raise ValueError("Source episode has no durable exchanges; cannot safely auto-resume")
        self.records = sorted(self.source.glob("[0-9]*.json")) if self.source else []
        if [p.name for p in self.records] != [f"{i:06d}.json" for i in range(1, len(self.records)+1)]:
            raise ValueError("Recovery journal is not contiguous")

    def __call__(self, endpoint, payload, headers):
        self.index += 1
        self.last_replayed = False
        name = f"{self.index:06d}.json"
        # Credentials deliberately excluded. Endpoint/model/input are pinned.
        fingerprint = digest({"endpoint": endpoint, "payload": payload})
        record = {"schema": 1, "turn": self.index, "request_sha256": fingerprint}
        if self.source and self.index <= len(self.records):
            saved = json.loads((self.source / name).read_text())
            if saved["request_sha256"] != fingerprint:
                raise ValueError(f"Recovery request mismatch at turn {self.index}; refusing new inference")
            if "response" in saved:
                atomic_json(self.directory / name, saved)
                self.replayed += 1
                self.last_replayed = True
                return saved["response"]
            if self.index != len(self.records):
                raise ValueError("Incomplete exchange precedes later responses")
        atomic_json(self.directory / name, record)
        atomic_json(self.directory.parent / "recovery-state.json",
                    {"status": "request_in_flight", "turn": self.index,
                     "delivery_may_be_ambiguous": True})
        try:
            response = self.transport(endpoint, payload, headers)
        except Exception as exc:
            atomic_json(self.directory.parent / "recovery-state.json",
                        {**failure_state(exc), "turn": self.index,
                         "delivery_may_be_ambiguous": not isinstance(exc, HTTPError)})
            raise
        # Commit the response BEFORE any of its tool calls can execute.
        record["response"] = response
        atomic_json(self.directory / name, record)
        atomic_json(self.directory.parent / "recovery-state.json",
                    {"status": "response_saved", "turn": self.index})
        return response


def restore_recorded_observations(session, source):
    """Verify rebuilt tool results/images, then retain original timing metadata."""
    from .harness import image_paths
    source = Path(source)
    events = []
    path = source / 'events.jsonl'
    if path.exists():
        lines = path.read_text().splitlines()
        for i, line in enumerate(lines):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                if i != len(lines)-1:
                    raise ValueError('Corrupt event journal') from None
                # A killed writer may leave an unfinished final line. Its tool
                # is reconstructed from the previously saved model response.
    execute = session.execute

    def stable(value, path=()):
        if isinstance(value, dict):
            return {k: stable(v, path + (k,)) for k, v in value.items()
                    if not k.endswith('_seconds')
                    and not (k in ('device', 'geometry_sources') and path[-1:] == ('rendering',))}
        if isinstance(value, list):
            return [stable(v, path) for v in value]
        if isinstance(value, str) and path[-1:] == ('image',):
            # Legacy mesh renders embed a wall-clock suffix in the filename.
            # Compare image bytes below; filenames are not observation content.
            return re.sub(r'^(frames/mesh-\d{6})-\d+(\.png)$', r'\1\2', value)
        return value

    def observation_difference(a, b, path='response'):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in a.keys() | b.keys():
                if key not in a or key not in b or a[key] != b[key]:
                    return observation_difference(a.get(key), b.get(key), path+'/'+key)
        if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            for i, (x, y) in enumerate(zip(a, b)):
                if x != y:return observation_difference(x, y, path+f'/{i}')
        return f'{path}: {type(a).__name__} {a!r} != {type(b).__name__} {b!r}'[:800]

    def checked(name, arguments=None, **kwargs):
        index = session.serial
        old = events[index] if index < len(events) else None
        if old and (old['tool'] != name or old['arguments'] != (arguments or {})):
            raise RuntimeError(f'Recovery tool mismatch at step {index+1}')
        response = execute(name, arguments, **kwargs)
        if old:
            if stable(response) != stable(old['response']):
                raise RuntimeError(f'Recovery observation mismatch at step {index+1}: '
                                   f'{observation_difference(stable(response), stable(old["response"]))}')
            for frame, current_frame in zip(image_paths(old['response']), image_paths(response), strict=True):
                a, b = (source / frame).resolve(), (session.output / current_frame).resolve()
                if not a.is_relative_to((source/'frames').resolve()) or not b.is_relative_to((session.output/'frames').resolve()):
                    raise RuntimeError('Invalid recorded image path')
                if a.read_bytes() != b.read_bytes():
                    # Mesa can round a tiny number of 3D edge pixels differently
                    # across processes. Never change the model's saved images:
                    # accept only bounded scene roundoff, then restore exact bytes.
                    from PIL import Image
                    import numpy as np
                    x, y = np.asarray(Image.open(a)), np.asarray(Image.open(b))
                    close = False
                    if (frame.endswith('-scene.png') or Path(frame).name.startswith('scene-')) and x.shape == y.shape and x.ndim == 3:
                        delta = np.abs(x.astype(np.int16)-y.astype(np.int16))
                        close = delta.max() <= 8 and np.mean(np.any(delta, axis=-1)) <= 1e-5
                    if not close:
                        raise RuntimeError(f'Recovery image mismatch at step {index+1}')
                    b.write_bytes(a.read_bytes())
                    with (session.output/'render-roundoff-restored.jsonl').open('a') as f:
                        f.write(json.dumps({'step':index+1,'image':frame,
                              'changed_pixels':int(np.any(delta,axis=-1).sum()),
                              'max_channel_delta':int(delta.max()),
                              'original_bytes_restored':True})+'\n')
                if current_frame != frame:
                    target_frame = (session.output / frame).resolve()
                    if not target_frame.is_relative_to((session.output/'frames').resolve()):
                        raise RuntimeError('Invalid recorded image path')
                    target_frame.write_bytes(a.read_bytes())
            # Keep the exact original tool text, including harmless timings, in
            # the model history. Newly executed actions use their new results.
            response = old['response']
            session.events[-1]['response'] = response
            target = session.output / 'events.jsonl'
            temp = target.with_suffix('.tmp')
            with temp.open('w') as f:
                for event in session.events:
                    f.write(json.dumps(event, allow_nan=False)+'\n')
                f.flush()
                os.fsync(f.fileno())
            temp.replace(target)
        return response

    session.execute = checked


def import_responses_episode(episode, config):
    """Import legacy stateless Responses logs using their ORIGINAL tool schemas.

    This reconstructs request fingerprints from recorded observations, not a
    provider-side request log. Recovery must still verify every regenerated
    request/image before it is allowed to make a new model call.
    """
    from .harness import content
    from .tools import schemas
    episode = Path(episode)
    model = config['model']
    if model['provider'] != 'responses' or config.get('demonstrations'):
        raise ValueError('Legacy import supports Responses episodes without demonstrations only')
    target = episode / 'exchanges'
    if target.exists():
        raise ValueError('Episode already has exchanges')
    read = lambda name: [json.loads(x) for x in (episode/name).read_text().splitlines() if x.strip()]
    events = {e['step']: e for e in read('events.jsonl')}
    actions = read('model-actions.jsonl')
    start = json.loads((episode/'initial-observations.json').read_text())[0]
    if events[start]['tool'] != 'look' or events[start-1]['tool'] != 'state':
        raise ValueError('Unrecognized initial observation layout')
    history = [{'role':'user', 'content':content({'state':events[start-1]['response'],
                'view':events[start]['response']}, episode, 'responses')}]
    system = (episode/'system-prompt.txt').read_text().removesuffix('\n')
    saved_options = json.loads((episode/'harness.json').read_text())['model_options']
    def request():
        return {'model':model['name'], 'instructions':system,
                'input':history, 'store':False, 'include':['reasoning.encrypted_content'],
                'max_output_tokens':config['budget']['output_tokens'], 'parallel_tool_calls':False,
                'tools':[{'type':'function','strict':False,**s} for s in schemas()], **saved_options}
    staging = episode/'exchanges-import.tmp'
    staging.mkdir(exist_ok=False)
    responses = read('model-responses.jsonl')
    for i, row in enumerate(responses, 1):
        if row['turn'] != i:
            raise ValueError('Legacy response turns are not contiguous')
        fingerprint = digest({'endpoint':model.get('endpoint','https://api.openai.com/v1/responses'), 'payload':request()})
        atomic_json(staging/f'{i:06d}.json', {'schema':1,'turn':i,'request_sha256':fingerprint,
                    'response':row['response'],'imported_from_legacy':True})
        history.extend(row['response'].get('output',[]))
        for a in actions:
            if a['turn'] != i:
                continue
            if a['step'] is None or a['step'] not in events:
                raise ValueError('Legacy action lacks a recorded tool response')
            history.append({'type':'function_call_output','call_id':a['call_id'],
                            'output':content(events[a['step']]['response'],episode,'responses')})
    # No inference is made by this import. The last request is pending.
    i = len(responses)+1
    atomic_json(staging/f'{i:06d}.json', {'schema':1,'turn':i,
                'request_sha256':digest({'endpoint':model.get('endpoint','https://api.openai.com/v1/responses'), 'payload':request()}),
                'imported_from_legacy':True})
    staging.rename(target)
    return {'saved_responses':len(responses), 'pending_turn':i, 'model_calls':0}
