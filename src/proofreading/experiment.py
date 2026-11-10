"""Portable, provider-independent evaluation; no scheduler or hosting dependency."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import time

from . import Dataset, Session
from .evaluate import evaluate_frozen, file_hash
from .harness import run
from .objectives import LOCAL_PROOFREADING

OPEN_WORLD_OBJECTIVE = """Reconstruct the starting selected process using EM evidence. Follow each branch
to an endpoint and find supported continuations. Inspect candidates for false merges
before joining them; preserve uncertain material and report unresolved endpoints.
Disconnected pieces can reconnect outside the crop: lack of in-crop connectivity
alone does not justify a split. Do not invent extensions outside the volume.
Preview and commit supported corrections, and finish with a concise account."""


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def load_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    allowed = {'schema', 'mode', 'model', 'budget', 'benchmark', 'phase', 'cases',
               'dataset', 'seed_segment', 'view', 'scene', 'scoring', 'objective', 'demonstrations'}
    if set(cfg) - allowed or cfg.get('schema') != 1:
        raise ValueError('Unknown configuration fields or schema (expected 1)')
    if cfg.get('mode') not in ('cb2', 'open-world'):
        raise ValueError('mode must be cb2 or open-world')
    model = cfg.get('model', {})
    if set(model) - {'provider', 'name', 'endpoint', 'options', 'api_key_env', 'transport', 'upload_directory'}:
        raise ValueError('Unknown model fields; credentials must be environment variables')
    if model.get('provider') not in ('responses', 'anthropic', 'chat', 'openrouter', 'gemini') or not model.get('name'):
        raise ValueError('Explicit model provider and name required')
    if model['provider'] == 'chat' and not model.get('endpoint'):
        raise ValueError('chat requires a full endpoint URL')
    if model.get('endpoint'):
        from urllib.parse import urlsplit
        url = urlsplit(model['endpoint'])
        if url.username or url.password or url.query or url.fragment or not url.hostname:
            raise ValueError('Endpoint must not contain credentials, query, or fragment')
        if url.scheme != 'https' and not (url.scheme == 'http' and url.hostname in ('127.0.0.1','localhost','::1')):
            raise ValueError('Endpoint must use HTTPS or loopback HTTP')
    if 'transport' in model or 'upload_directory' in model:
        if (model.get('transport') != 'moonshot-files' or model['provider'] != 'chat'
                or model['name'] != 'kimi-k3'
                or model.get('endpoint') != 'https://api.moonshot.ai/v1/chat/completions'
                or not isinstance(model.get('upload_directory'), str) or not model['upload_directory']):
            raise ValueError('moonshot-files requires kimi-k3, the native endpoint, and upload_directory')
        model['upload_directory'] = str((path.parent / model['upload_directory']).resolve())
    budget = cfg.setdefault('budget', {})
    if set(budget) - {'actions', 'turns', 'output_tokens'}:
        raise ValueError('Unknown budget fields')
    for key, default in [('actions', 40), ('turns', 40), ('output_tokens', 8192)]:
        budget.setdefault(key, default)
        if type(budget[key]) is not int or budget[key] < 1:
            raise ValueError('Budgets must be positive integers')
    def resolve(value):
        return str((path.parent / value).resolve())
    if cfg.get('demonstrations'):
        cfg['demonstrations'] = resolve(cfg['demonstrations'])
    if cfg['mode'] == 'cb2':
        if cfg.get('phase') not in ('train', 'test'):
            raise ValueError('CB2 phase must explicitly be train or test')
        if not cfg.get('cases') or len(set(cfg['cases'])) != len(cfg['cases']):
            raise ValueError('Select explicit unique CB2 case IDs')
        cfg['benchmark'] = resolve(cfg['benchmark'])
    else:
        cfg['dataset'] = resolve(cfg['dataset'])
        if not isinstance(cfg.get('seed_segment'), str) or not cfg['seed_segment'].isdigit():
            raise ValueError('Open-world seed_segment must be a decimal string')
        scoring = cfg.get('scoring', {})
        if scoring.get('kind') == 'frozen':
            if set(scoring) != {'kind', 'spec'}:
                raise ValueError('frozen scoring requires spec')
            scoring['spec'] = resolve(scoring['spec'])
        elif scoring.get('kind') == 'transfer':
            if set(scoring) != {'kind', 'reference', 'scope'}:
                raise ValueError('transfer scoring requires reference and scope')
            for key in ('reference', 'scope'):
                scoring[key] = resolve(scoring[key])
        else:
            raise ValueError('Open-world scoring must be frozen or transfer; no silent unscored run')
    return cfg


def cases(cfg):
    if cfg['mode'] == 'open-world':
        return [{'case_id': 'open-world', 'dataset': cfg['dataset']}]
    root = Path(cfg['benchmark'])
    records = json.loads((root / 'index.json').read_text())
    selected = [r for r in records if r['case_id'] in cfg['cases']]
    if len(selected) != len(cfg['cases']) or {r['case_id'] for r in selected} != set(cfg['cases']):
        raise ValueError('Unknown or duplicate CB2 case ID')
    result = []
    for r in selected:
        if r['split'] != cfg['phase']:
            raise ValueError('CB2 case does not belong to the requested split')
        case = (root / r['case_id']).resolve()
        if not case.is_relative_to(root) or Path(r['case_id']).name != r['case_id']:
            raise ValueError('Invalid case path')
        result.append({**r, 'dataset': str(case / 'public/dataset.json'), 'directory': str(case)})
    return result


def scorer_for(cfg, record, dataset):
    if cfg['mode'] == 'cb2':
        if dataset.fingerprint != record['dataset_sha256']:
            raise ValueError('Frozen dataset changed')
        directory = Path(record['directory']) / 'evaluator'
        names = record.get('scorers', ['human', 'certified'])
        if names not in (['human'], ['human', 'certified']):
            raise ValueError('Invalid CB2 scoring contracts')
        if names == ['human'] and not record.get('certified_unavailable'):
            raise ValueError('Missing certified scoring requires an explicit coverage explanation')
        evaluator = evaluate_frozen
        if getattr(dataset,'partition_format',None)=='streaming-rules-v1':
            from .streamed_evaluate import evaluate_streamed
            evaluator = evaluate_streamed
        def evaluate(episode):
            scores = {name: evaluator(dataset, episode, directory / (name + '-frozen.json'))
                      for name in names}
            if record.get('snapshot_reference'):
                reference = directory / 'snapshot-frozen.json'
                if file_hash(reference) != record['snapshot_reference']:
                    raise ValueError('Snapshot scoring contract changed')
                contract = json.loads(reference.read_text())
                if contract.get('scope_kind') != 'full-cube':
                    raise ValueError('Legacy target-only snapshot cannot score wrong in-cube additions. '
                                     'Run connectome-bundle upgrade-reference before inference.')
                from .release_scoring import evaluate_volume
                scores['snapshot'] = evaluate_volume(episode, reference)
            return scores
        return evaluate
    scoring = cfg['scoring']
    if scoring['kind'] == 'frozen':
        if getattr(dataset,'partition_format',None)=='streaming-rules-v1':
            from .streamed_evaluate import evaluate_streamed
            return lambda episode: {'frozen':evaluate_streamed(dataset,episode,scoring['spec'])}
        return lambda episode: {'frozen': evaluate_frozen(dataset, episode, scoring['spec'])}
    from .indexed_evaluate import CertifiedScorer
    from .transfer_evaluate import evaluate_transfer
    reference, scope = Path(scoring['reference']), Path(scoring['scope'])
    spec = json.loads((scope / 'frozen.json').read_text())
    if str(spec['seed_segment']) != cfg['seed_segment']:
        raise ValueError('Public starting seed does not match frozen transfer scope')
    scorer = CertifiedScorer(dataset, reference)
    scorer.load_baseline(reference / 'baseline.npy')
    return lambda episode: {'transfer': evaluate_transfer(dataset, scorer, scope, episode)}


def setup(cfg, record, session):
    if cfg['mode'] == 'cb2':
        task = json.loads((Path(record['directory']) / 'public/task.json').read_text())
        seed = task['seed_segment']
        view = {'center_nm': task['center_nm'], 'pixel_size_nm': 16, 'width': 384, 'height': 384, **cfg.get('view', {})}
        scene = {'target_nm': task['center_nm'], 'view_height_nm': task['extent_nm'] * 1.1, **cfg.get('scene', {})}
        if hasattr(getattr(session,'dataset',None),'initial_view'):
            view={**session.dataset.initial_view,'center_nm':task['center_nm'],**cfg.get('view',{})}
            scene={'target_nm':task['center_nm'],'view_height_nm':view['pixel_size_nm']*view['height']*1.2,**cfg.get('scene',{})}
    else:
        seed = cfg['seed_segment']
        import numpy as np
        dataset = session.dataset
        if hasattr(dataset, 'to_dense'):
            atoms = np.asarray(sorted(dataset.initial.atoms_for(int(seed))), dtype=np.int64)
            dense = dataset.to_dense(atoms)
            largest = dense[np.argmax(dataset.atom_cnt[dense])]
            center = dataset.atoms.origin + (dataset.atom_cent_vox[largest] + .5) * dataset.atoms.resolution
        elif hasattr(dataset,'initial_view'):
            center=np.asarray(dataset.initial_view['center_nm'])
        else:
            atoms = [a for a, s in dataset.initial.items() if s == int(seed)]
            points = np.argwhere(np.isin(dataset.atoms.data, atoms))
            if not len(points):
                raise ValueError('Starting segment is absent')
            center = dataset.atoms.origin + (points[len(points)//2] + .5) * dataset.atoms.resolution
        view = {'center_nm': center.tolist(), 'pixel_size_nm': 16, 'width': 512, 'height': 512, **cfg.get('view', {})}
        scene = {'target_nm': center.tolist(), 'view_height_nm': 22000, **cfg.get('scene', {})}
    # Source resolution has its own tool schema; set_view only changes display.
    resolution = view.pop('em_resolution_nm', None)
    actions = [('select', {'segments': [str(seed)], 'mode': 'replace'}), ('set_view', view)]
    if resolution is not None:
        actions.append(('set_em_resolution', {'xy_nm': resolution}))
    actions.append(('set_scene', scene))
    for name, arguments in actions:
        result = session.execute(name, arguments)
        if not result['ok']:
            raise ValueError('Initial setup failed: ' + str(result))


def _execute(config, output, preflight=False, shard_index=0, shard_count=1, transport=None, resume_from=None):
    cfg = load_config(config)
    source = Path(resume_from) if resume_from else None
    if source and json.loads((source / 'config.json').read_text()) != cfg:
        raise ValueError('Resume requires identical configuration; do not silently change the experiment')
    from .runtime import configure
    configure()
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError('Invalid shard index/count')
    selected = cases(cfg)[shard_index::shard_count]
    if not selected:
        raise ValueError('Shard has no cases')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)  # Never silently repeat a paid run.
    write(output / 'config.json', cfg)
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = os.environ.get('CONNECTOME_SOURCE_REVISION', 'unavailable')
    from .storage import path_map
    write(output / 'run.json', {'schema': 1, 'config_sha256': file_hash(config), 'source_commit': commit,
                              'python': platform.python_version(), 'platform': platform.platform(),
                              'path_map': path_map(), 'world_root': os.environ.get('CONNECTOME_WORLD_ROOT'),
                              'preflight': preflight, 'shard_index': shard_index, 'shard_count': shard_count,
                              'created_at_unix': time.time()})
    results = []
    for record in selected:
        dest = output / record['case_id']
        previous = source / record['case_id'] if source else None
        if previous and (previous / 'result.json').exists():
            import shutil
            shutil.copytree(previous, dest)
            result = json.loads((dest / 'result.json').read_text())
            results.append({'case_id': record['case_id'], **result})
            continue
        dest.mkdir()
        write(dest / 'intent.json', {'mode': cfg['mode'], 'case_id': record['case_id'], 'model': cfg['model'],
                                    'budget': cfg['budget'], 'preflight': preflight})
        session = scorer = dataset = None
        try:
            dataset = Dataset.load(record['dataset'])
            session_options = {}
            if previous and (previous / 'episode/session.json').exists():
                meta = json.loads((previous / 'episode/session.json').read_text())
                if meta['dataset_sha256'] != dataset.fingerprint:
                    raise ValueError('Resume dataset fingerprint mismatch')
                session_options = {'seeded_split_policy': meta.get('seeded_split_policy', 'connected_components'),
                                   'split_algorithm': meta.get('split_algorithm', 'min_cut')}
            # Permit the pinned older runtime whose only splitter was min_cut.
            if session_options:
                import inspect
                if 'split_algorithm' not in inspect.signature(Session).parameters:
                    if session_options.pop('split_algorithm') != 'min_cut':
                        raise ValueError('Pinned runtime lacks the recorded splitter')
            session = Session(dataset, dest / 'episode', **session_options)
            if previous and (previous / 'episode').exists():
                from .recovery import restore_recorded_observations
                restore_recorded_observations(session, previous / 'episode')
            scorer = scorer_for(cfg, record, dataset)
            initial = scorer(session.output)  # Verify frozen reference before any inference.
            write(dest / 'initial-score.json', initial)
            setup(cfg, record, session)
            if preflight:
                result = {'status': 'preflight', 'model_calls': 0, 'initial_scores': initial,
                          'dataset_sha256': dataset.fingerprint}
            else:
                model, budget = cfg['model'], cfg['budget']
                kwargs = {'transport': transport} if transport is not None else {}
                if transport is None and model.get('transport') == 'moonshot-files':
                    from .moonshot_transport import MoonshotTransport
                    kwargs['transport'] = MoonshotTransport(model['upload_directory'])
                if previous and (previous / 'episode/exchanges').exists():
                    kwargs['resume_from'] = previous / 'episode'
                elif previous and (previous / 'episode/model-responses.jsonl').exists():
                    raise ValueError('Legacy episode needs an audited journal import before recovery')
                if model.get('api_key_env'):
                    kwargs['api_key'] = os.environ.get(model['api_key_env'], '')
                summary = run(session, model['name'], provider=model['provider'],
                              max_turns=budget['turns'], max_actions=budget['actions'],
                              max_output_tokens=budget['output_tokens'], endpoint=model.get('endpoint'),
                              model_options=model.get('options'),
                              objective=cfg.get('objective', LOCAL_PROOFREADING if cfg['mode'] == 'cb2' else OPEN_WORLD_OBJECTIVE),
                              demonstrations=cfg.get('demonstrations'), **kwargs)
                result = {'status': summary['status'], 'harness': summary, 'scores': scorer(session.output),
                          'quality_flags': ['early_model_stop'] if summary['status'] == 'model_stopped_without_finish' else []}
            write(dest / 'result.json', result)
            results.append({'case_id': record['case_id'], **result})
            print(json.dumps({'case_id': record['case_id'], 'status': result['status']}), flush=True)
        except Exception as exc:
            from .recovery import failure_state
            failure = failure_state(exc)
            def record_failure(path, value):
                try:
                    write(path, value)
                except OSError as logging_error:
                    # Preserve the original failure when quotas also prevent
                    # checkpoint writes; scheduler/container logs remain useful.
                    import sys
                    print(json.dumps({'failure': failure, 'recording_error': type(logging_error).__name__}),
                          file=sys.stderr, flush=True)
            record_failure(dest / 'failure.json', failure)
            # Expose partial scores as partial, never as a completed evaluation.
            if session is not None and scorer is not None:
                try:
                    write(dest / 'partial-result.json', {'status': 'paused', 'complete': False,
                          'failure': failure, 'scores': scorer(session.output)})
                except Exception as score_error:
                    record_failure(dest / 'partial-score-failure.json', {'error_type': type(score_error).__name__})
            record_failure(output / 'status.json', {**failure, 'case_id': record['case_id']})
            raise
        finally:
            if dataset is not None and callable(getattr(dataset, 'close', None)):
                dataset.close()
    write(output / 'results.json', results)
    write(output / 'status.json', {'status': 'completed', 'cases': len(results)})
    return results


def execute(config, output, preflight=False, shard_index=0, shard_count=1, transport=None, resume_from=None):
    # Locks prevent simultaneous writers or recovery from an actively running source.
    import fcntl
    from contextlib import ExitStack
    with ExitStack() as stack:
        for directory in ([Path(resume_from)] if resume_from else []) + [Path(output)]:
            directory.parent.mkdir(parents=True, exist_ok=True)
            lock = stack.enter_context((directory.parent / (directory.name + '.lock')).open('a'))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError('Run or recovery already active') from None
        return _execute(config, output, preflight, shard_index, shard_count, transport, resume_from)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--preflight', action='store_true', help='Validate data, scoring and rendering without model calls')
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--shard-count', type=int, default=1)
    p.add_argument('--resume-from', type=Path, help='Recover into a new output directory, reusing saved model responses')
    a = p.parse_args()
    execute(a.config, a.out, a.preflight, a.shard_index, a.shard_count, resume_from=a.resume_from)


if __name__ == '__main__':
    main()
