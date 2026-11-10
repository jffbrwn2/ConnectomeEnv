import pytest
from proofreading.transfer_evaluate import choose_anchor


def test_anchor_uses_certified_seed_volume_and_stable_integer_tie_break():
    high=2**56
    assert choose_anchor([high+2,high+1,high+3],[7,7,8],[100,100,999],7)==high+1
    assert choose_anchor([high+1,high+2],[7,7],[100,101],7)==high+2
    with pytest.raises(ValueError):choose_anchor([1],[8],[9],7)

from test_indexed_evaluate import scorer_case


def test_frozen_transfer_tracks_seed_and_reports_wrong_and_unknown_growth(scorer_case):
    import json,time
    import numpy as np
    from proofreading.evaluate import file_hash
    from proofreading.transfer_evaluate import freeze_anchor,evaluate_transfer
    scorer,atoms,d,path=scorer_case
    d.roots=np.array([30,31]);d.root_ip=np.array([0,1,4]);d.root_atoms=atoms;d.unknown_ownership=set()
    scorer.build_baseline()
    scope=path/'scope';scope.mkdir()
    np.save(scope/'seed-atoms.npy',atoms[:1]);np.save(scope/'reachable-atoms.npy',atoms[:2])
    spec={'dataset_sha256':d.fingerprint,'reference_spec_sha256':scorer.spec_hash,
          'seed_atoms_sha256':file_hash(scope/'seed-atoms.npy'),'reachable_atoms_sha256':file_hash(scope/'reachable-atoms.npy'),
          'seed_segment':'30','evaluator_target_neuron':'11','frozen_at_unix':time.time(),'scope':'synthetic crop graph'}
    (scope/'frozen.json').write_text(json.dumps(spec));contract=freeze_anchor(d,scorer,scope)
    episode=path/'episode';episode.mkdir()
    meta={'dataset_sha256':d.fingerprint,'created_at_unix':contract['created_at_unix']+1}
    (episode/'session.json').write_text(json.dumps(meta))
    np.savez(episode/'partition.npz',atoms=atoms[1:],segments=np.full(3,30,dtype=np.uint64),format='sparse-overlay-v1',dataset_sha256=d.fingerprint)
    result=evaluate_transfer(d,scorer,scope,episode)
    assert result['evidence']['recall']['initial']==3/8
    assert result['evidence']['recall']['final']==1
    assert result['evidence']['added']['target_voxels']==5
    assert result['evidence']['added']['other_certified_neuron_voxels']==2
    assert result['evidence']['added']['unknown_voxels']==7
    meta['created_at_unix']=0;(episode/'session.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='before inference'):evaluate_transfer(d,scorer,scope,episode)
