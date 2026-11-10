"""Evaluate a streamed episode on an independently frozen local reference.

Exploration never expands the scoring denominator. Material outside this local
reference remains unscored; no whole-neuron correctness claim is inferred.
"""
import json,time
from pathlib import Path
import numpy as np
from .evaluate import file_hash,metrics,scope_attachment


def freeze_streamed(dataset, core_manifest, original_spec, output):
    from .volume import Dataset
    core_manifest=Path(core_manifest).resolve();original_spec=Path(original_spec).resolve()
    core=Dataset.load(core_manifest);old=json.loads(original_spec.read_text())
    if old['dataset_sha256']!=core.fingerprint:raise ValueError('Original frozen case changed')
    for name in ('reference','scope'):
        if file_hash(old[name])!=old[name+'_sha256']:raise ValueError('Original reference changed')
    owners=dataset.source.ensure(np.asarray(list(core.initial),np.uint64))
    if owners!=core.initial:raise ValueError('Streaming timestamp does not reproduce the benchmark initial partition')
    spec={'schema':'streamed-frozen-v1','dataset_sha256':dataset.fingerprint,
          'core_manifest':str(core_manifest),'core_sha256':core.fingerprint,
          'original_spec':str(original_spec),'original_spec_sha256':file_hash(original_spec),
          'frozen_at_unix':time.time(),
          'scope_note':'Original local reference only. Exploration outside it is unscored, not automatically correct.',
          **{k:old[k] for k in ('reference','reference_sha256','scope','scope_sha256')}}
    return _pack_spec(core, spec, output)


def project_rules(initial, rules):
    """Apply global identity rules to the fixed reference atoms, without imagery."""
    partition=dict(initial)
    retired=set();children_used=set()
    for rule in rules:
        if set(rule)!={'parents','children','assignments'}:raise ValueError('Invalid streamed edit rule')
        parents=set(map(int,rule['parents']));children=list(map(int,rule['children']))
        if not parents or not children or len(set(children))!=len(children) or any(c<=0 for c in children):
            raise ValueError('Invalid streamed edit identities')
        if parents&retired or set(children)&(parents|retired|children_used):raise ValueError('Reused/retired segment identity')
        assignments=rule['assignments']
        if assignments is None:
            if len(parents)<2 or len(children)!=1:raise ValueError('Invalid merge rule')
        else:
            if len(parents)!=1 or len(children)<2:raise ValueError('Invalid split rule')
            assignments={int(a):int(s) for a,s in assignments.items()}
            if not assignments or any(a<=0 or s not in children for a,s in assignments.items()):raise ValueError('Invalid split assignments')
        for atom,sid in partition.items():
            if sid not in parents:continue
            if assignments is not None and atom not in assignments:raise ValueError('Split omitted a reference atom')
            partition[atom]=children[0] if assignments is None else assignments[atom]
        retired.update(parents);children_used.update(children)
    return partition


def _evaluate_legacy(dataset, episode, spec_file):
    from .volume import Dataset
    episode=Path(episode);spec=json.loads(Path(spec_file).read_text())
    meta=json.loads((episode/'session.json').read_text())
    if spec.get('schema')!='streamed-frozen-v1' or spec['dataset_sha256']!=dataset.fingerprint or meta['dataset_sha256']!=dataset.fingerprint:
        raise ValueError('Streamed scoring identity mismatch')
    if spec['frozen_at_unix']>meta['created_at_unix']:raise ValueError('Scoring was not frozen before the episode')
    for key in ('original_spec','reference','scope'):
        if file_hash(spec[key])!=spec[key+'_sha256']:raise ValueError('Frozen scoring file changed')
    core=Dataset.load(spec['core_manifest'])
    if core.fingerprint!=spec['core_sha256']:raise ValueError('Frozen core dataset changed')
    with np.load(episode/'partition.npz',allow_pickle=False) as z:
        if z['format'].item()!='streaming-rules-v1' or z['dataset_sha256'].item()!=dataset.fingerprint:
            raise ValueError('Streamed partition identity mismatch')
        rules=json.loads(z['rules'].item())
    final=project_rules(core.initial,rules)
    reference=np.load(spec['reference'],mmap_mode='r');scope=np.load(spec['scope'],mmap_mode='r')
    initial_metrics=metrics(core.atoms.data,core.initial,reference,scope)
    final_metrics=metrics(core.atoms.data,final,reference,scope)
    return {'initial':initial_metrics,'final':final_metrics,
            'vi_improvement_bits':initial_metrics['vi_total_bits']-final_metrics['vi_total_bits'],
            'scope_attachment':scope_attachment(core.atoms.data,core.initial,final,scope,atom_counts=(core.ids,core.counts)),
            'scope_note':spec['scope_note'],'outside_reference_material':'unscored',
            'reference_visible_to_agent':False,'frozen_before_episode':True,
            'frozen_spec_sha256':file_hash(spec_file),'dataset_sha256':dataset.fingerprint}


def _pack_spec(core, legacy, output):
    from .compact_scoring import pack
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    counts=output.with_suffix('.counts.npz')
    pack(core.atoms.data,core.initial,np.load(legacy['reference'],mmap_mode='r'),
         np.load(legacy['scope'],mmap_mode='r'),counts)
    spec={'schema':'streamed-frozen-v2','dataset_sha256':legacy['dataset_sha256'],
          'counts':counts.name,'counts_sha256':file_hash(counts),
          'frozen_at_unix':legacy['frozen_at_unix'],'packaged_at_unix':time.time(),
          'scope_note':legacy['scope_note'],
          'provenance':{k:legacy[k] for k in ('core_sha256','original_spec_sha256','reference_sha256','scope_sha256')}}
    if 'legacy_contract_sha256' in legacy:spec['provenance']['legacy_contract_sha256']=legacy['legacy_contract_sha256']
    with output.open('x') as f:f.write(json.dumps(spec,indent=2)+'\n')
    return spec


def compact_contract(spec_file, output):
    """One-time verified migration; inherit the original freeze, not a new scope."""
    from .volume import Dataset
    path=Path(spec_file).resolve();spec=json.loads(path.read_text())
    if spec.get('schema')!='streamed-frozen-v1':raise ValueError('Expected legacy streamed scoring contract')
    for key in ('original_spec','reference','scope'):
        if file_hash(spec[key])!=spec[key+'_sha256']:raise ValueError('Frozen scoring file changed')
    core=Dataset.load(spec['core_manifest'])
    if core.fingerprint!=spec['core_sha256']:raise ValueError('Frozen core dataset changed')
    spec['legacy_contract_sha256']=file_hash(path)
    return _pack_spec(core,spec,output)


def evaluate_package(episode, spec_file, *, expected_fingerprint=None):
    """Offline scoring requires only the package and saved episode state."""
    from .compact_scoring import read,score
    episode=Path(episode);path=Path(spec_file).resolve();spec=json.loads(path.read_text())
    if spec.get('schema')!='streamed-frozen-v2':raise ValueError('Expected compact scoring package')
    meta=json.loads((episode/'session.json').read_text());fingerprint=spec['dataset_sha256']
    if meta['dataset_sha256']!=fingerprint or (expected_fingerprint is not None and expected_fingerprint!=fingerprint):
        raise ValueError('Streamed scoring identity mismatch')
    if spec['frozen_at_unix']>meta['created_at_unix']:raise ValueError('Scoring was not frozen before the episode')
    name=spec['counts']
    if Path(name).name!=name:raise ValueError('Counts must be packaged beside the contract')
    counts=path.parent/name
    if file_hash(counts)!=spec['counts_sha256']:raise ValueError('Frozen scoring counts changed')
    data=read(counts);initial={int(a):int(s) for a,s in zip(data['atoms'],data['initial'])}
    with np.load(episode/'partition.npz',allow_pickle=False) as z:
        if z['format'].item()!='streaming-rules-v1' or z['dataset_sha256'].item()!=fingerprint:
            raise ValueError('Streamed partition identity mismatch')
        rules=json.loads(z['rules'].item())
    result=score(data,initial,project_rules(initial,rules))
    return {**result,'scope_note':spec['scope_note'],'outside_reference_material':'unscored',
            'reference_visible_to_agent':False,'frozen_before_episode':True,
            'frozen_spec_sha256':file_hash(path),'dataset_sha256':fingerprint}


def evaluate_streamed(dataset, episode, spec_file):
    schema=json.loads(Path(spec_file).read_text()).get('schema')
    if schema=='streamed-frozen-v2':
        return evaluate_package(episode,spec_file,expected_fingerprint=dataset.fingerprint)
    return _evaluate_legacy(dataset,episode,spec_file)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description='Package legacy scoring once, or score offline without loading an environment')
    commands=p.add_subparsers(dest='command',required=True)
    c=commands.add_parser('compact');c.add_argument('spec');c.add_argument('output')
    c=commands.add_parser('score');c.add_argument('episode');c.add_argument('spec')
    args=p.parse_args()
    result=compact_contract(args.spec,args.output) if args.command=='compact' else evaluate_package(args.episode,args.spec)
    print(json.dumps(result,indent=2))
