"""Precommitted anchor-based transfer evaluation; never exposed to agent tools."""
import json,time
from pathlib import Path
import numpy as np
from .evaluate import file_hash
from .indexed_evaluate import load_partition,extension_evidence


def _scope(dataset,scorer,directory):
    directory=Path(directory);spec=json.loads((directory/'frozen.json').read_text())
    if spec['dataset_sha256']!=dataset.fingerprint or spec['reference_spec_sha256']!=scorer.spec_hash:
        raise ValueError('Transfer reference or dataset changed')
    arrays={}
    for name in ('seed','reachable'):
        path=directory/(name+'-atoms.npy')
        if file_hash(path)!=spec[name+'_atoms_sha256']:raise ValueError('Frozen transfer atoms changed')
        arrays[name]=np.load(path,allow_pickle=False)
    return spec,arrays


def choose_anchor(atoms,reference,counts,target):
    atoms=np.asarray(atoms);reference=np.asarray(reference);counts=np.asarray(counts)
    if atoms.ndim!=1 or atoms.shape!=reference.shape or atoms.shape!=counts.shape:
        raise ValueError('Anchor arrays must align')
    valid=np.flatnonzero(reference==target)
    if not len(valid):raise ValueError('Seed lacks certified target material')
    # Deterministic under input ordering and label values above floating precision.
    index=min(valid,key=lambda i:(-int(counts[i]),int(atoms[i])))
    return int(atoms[index])


def freeze_anchor(dataset,scorer,directory):
    directory=Path(directory);spec,arrays=_scope(dataset,scorer,directory)
    refs,counts,_,_=scorer.lookup(arrays['seed'])
    atom=choose_anchor(arrays['seed'],refs,counts,int(spec['evaluator_target_neuron']))
    if dataset.initial[atom]!=int(spec['seed_segment']):raise ValueError('Anchor is not in the initial seed')
    contract={'schema':1,'created_at_unix':time.time(),'anchor_atom':str(atom),
              'scope_sha256':file_hash(directory/'frozen.json'),
              'selection_rule':'Largest certified target atom in initial seed; ties use lower native atom ID. Score the final segment containing this fixed atom, irrespective of final selection.'}
    with (directory/'anchor.json').open('x') as f:f.write(json.dumps(contract,indent=2)+'\n')
    return contract


def evaluate_transfer(dataset,scorer,directory,episode):
    directory=Path(directory);episode=Path(episode)
    spec,arrays=_scope(dataset,scorer,directory)
    contract=json.loads((directory/'anchor.json').read_text())
    meta=json.loads((episode/'session.json').read_text())
    if meta['dataset_sha256']!=dataset.fingerprint:raise ValueError('Episode dataset mismatch')
    if contract['scope_sha256']!=file_hash(directory/'frozen.json'):raise ValueError('Transfer scope changed')
    if max(spec['frozen_at_unix'],contract['created_at_unix'])>meta['created_at_unix']:
        raise ValueError('Transfer contract was not frozen before inference')
    initial_refs,initial_counts,_,_=scorer.lookup(arrays['seed'])
    expected=choose_anchor(arrays['seed'],initial_refs,initial_counts,int(spec['evaluator_target_neuron']))
    if int(contract['anchor_atom'])!=expected:raise ValueError('Anchor differs from deterministic selection')
    partition=load_partition(dataset,episode/'partition.npz')
    segment=partition[expected];final=partition.atoms_for(segment)
    all_atoms=np.asarray(sorted(set(map(int,arrays['seed']))|final|set(map(int,arrays['reachable']))),dtype=np.uint64)
    refs,counts,_,_=scorer.lookup(all_atoms)
    evidence=extension_evidence(all_atoms,refs,counts,arrays['seed'],final,int(spec['evaluator_target_neuron']),arrays['reachable'])
    return {'anchor_atom':str(expected),'final_segment':str(segment),'anchor_contract_sha256':file_hash(directory/'anchor.json'),
            'frozen_before_episode':True,'evidence':evidence,'recall_scope':spec['scope'],
            'full_neuron_recall':False,'global_identity_diagnostics':scorer.score(partition)}
