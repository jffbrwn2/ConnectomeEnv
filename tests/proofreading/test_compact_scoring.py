import numpy as np
from proofreading.compact_scoring import pack,read,score
from proofreading.evaluate import metrics,scope_attachment


def test_compact_matches_dense_with_mixed_reference_background_unknown_and_attachment(tmp_path):
    rng=np.random.default_rng(7)
    atoms=rng.integers(0,6,size=(17,19,7),dtype=np.uint64)
    ref=rng.integers(0,4,size=atoms.shape,dtype=np.uint64)
    ref[ref>0]+=2**63
    scope=rng.random(atoms.shape)>.3
    initial={a:2**63+a for a in range(1,6)}
    final={a:2**63+(a%2)+100 for a in range(1,6)}
    path=tmp_path/'counts.npz';pack(atoms,initial,ref,scope,path)
    result=score(read(path),initial,final)
    assert result['initial']==metrics(atoms,initial,ref,scope)
    assert result['final']==metrics(atoms,final,ref,scope)
    assert result['scope_attachment']==scope_attachment(atoms,initial,final,scope)


def test_all_target_and_partial_atom_outside_counts(tmp_path):
    atoms=np.array([1,1,2,2,3],np.uint64).reshape(5,1,1)
    initial={1:10,2:20,3:30};final={1:40,2:40,3:30}
    ref=np.array([5,5,6,0,0],np.uint64).reshape(atoms.shape)
    for i,scope in enumerate([np.ones_like(atoms,bool),np.array([1,0,0,0,0],bool).reshape(atoms.shape)]):
        path=tmp_path/f'{i}.npz';pack(atoms,initial,ref,scope,path)
        result=score(read(path),initial,final)
        assert result['final']==metrics(atoms,final,ref,scope)
        assert result['scope_attachment']==scope_attachment(atoms,initial,final,scope)
