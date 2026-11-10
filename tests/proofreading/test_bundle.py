import io
import json
import tarfile
from pathlib import Path

import pytest
from proofreading.bundle import verify, unpack
from proofreading.evaluate import file_hash


def make_bundle(p):
    p.mkdir();(p/'index.json').write_text('[]')
    (p/'bundle.json').write_text(json.dumps({'schema':'connectome-bundle-v1','cases':0,'files':{'index.json':file_hash(p/'index.json')}}))


def test_roundtrip_and_tampering(tmp_path):
    src=tmp_path/'source';make_bundle(src)
    archive=tmp_path/'bundle.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        for p in src.iterdir():t.add(p,arcname=p.name)
    out=tmp_path/'out'
    assert unpack(archive,out)['status']=='verified'
    (out/'index.json').write_text('[1]')
    with pytest.raises(ValueError,match='changed'):verify(out)


@pytest.mark.parametrize('name,link',[('../escape',False),('/tmp/escape',False),('link',True)])
def test_archive_rejects_escape_and_links(tmp_path,name,link):
    archive=tmp_path/'bad.tar'
    with tarfile.open(archive,'w') as t:
        i=tarfile.TarInfo(name)
        if link:i.type=tarfile.SYMTYPE;i.linkname='/etc/passwd'
        t.addfile(i,io.BytesIO(b''))
    with pytest.raises(ValueError):unpack(archive,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_export_preserves_identity_and_portable_scoring(tmp_path):
    import hashlib
    import numpy as np
    from types import SimpleNamespace
    from test_snapshot_scoring import fixture
    from proofreading.snapshot_scoring import prepare
    from proofreading.bundle import export
    from proofreading.release_scoring import evaluate_volume
    data, initial, old = fixture(tmp_path)
    dataset={'backend':'streamed-world-v1','name':'fixture','cache_directory':'/private/cluster/cache'}
    identity={k:v for k,v in dataset.items() if k!='cache_directory'}
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    compact=json.loads(old.read_text());compact['dataset_sha256']=fingerprint;old.write_text(json.dumps(compact))
    ref=tmp_path/'snapshot.json'
    client=SimpleNamespace(datastack_name='test',chunkedgraph=SimpleNamespace(
        get_roots=lambda atoms,timestamp:np.array([20,20,30,30],np.uint64)))
    prepare(old,ref,client,'2026-09-28T00:00:00+00:00')
    bench=tmp_path/'benchmark';case=bench/'case';(case/'public').mkdir(parents=True);(case/'evaluator').mkdir()
    (case/'public/dataset.json').write_text(json.dumps(dataset));(case/'public/task.json').write_text('{}')
    (case/'evaluator/human-frozen.json').write_text(old.read_text())
    (case/'evaluator/old.npz').write_bytes((tmp_path/'old.npz').read_bytes())
    core=tmp_path/'core';(core/'public').mkdir(parents=True);(core/'evaluator').mkdir()
    (core/'public/dataset.json').write_text(json.dumps({'atoms':{'resolution_nm':[8,8,40]}}))
    np.savez(core/'evaluator/before.npz',atoms=np.arange(1,6,dtype=np.uint64),segments=np.full(5,10,dtype=np.uint64))
    np.savez(core/'evaluator/after.npz',atoms=np.arange(1,6,dtype=np.uint64),segments=np.array([90,90,91,91,99],np.uint64))
    (case/'evaluator/provenance.json').write_text(json.dumps({'source_case':str(core)}))
    (bench/'index.json').write_text(json.dumps([{'case_id':'case','split':'test','dataset_sha256':fingerprint,'scorers':['human'],'certified_unavailable':'fixture','source_case':str(core)}]))
    run=tmp_path/'run';run.mkdir();(run/'config.json').write_text(json.dumps({'benchmark':str(bench),'cases':['case']}))
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'jobs':[{'root':str(run),'reference':str(ref),'kind':'split_edit'}]}))
    out=tmp_path/'portable';assert export(manifest,out)['cases']==1
    assert 'source_case' not in json.loads((out/'index.json').read_text())[0]
    ep=tmp_path/'ep';ep.mkdir();(ep/'session.json').write_text(json.dumps({'dataset_sha256':fingerprint,'created_at_unix':1}))
    np.savez(ep/'partition.npz',format='streaming-rules-v1',dataset_sha256=fingerprint,rules='[]')
    v=evaluate_volume(ep,out/'case/evaluator/snapshot-frozen.json')['volume']
    assert v['initial_error_voxels']==2 and v['human_error_voxels']==0


def test_verify_rejects_unlisted_symlink_directory(tmp_path):
    src=tmp_path/'source';make_bundle(src)
    (src/'escape').symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises(ValueError,match='Symlinks'):verify(src)


def test_upgrade_reference_covers_surroundings_and_preserves_source(tmp_path):
    from types import SimpleNamespace
    import numpy as np
    from test_snapshot_scoring import fixture
    from proofreading.snapshot_scoring import prepare
    from proofreading.bundle import upgrade_reference
    from proofreading.compact_scoring import read
    src=tmp_path/'source';make_bundle(src)
    d=src/'case/evaluator';d.mkdir(parents=True)
    _,_,old=fixture(d)
    old.rename(d/'human-frozen.json')
    client=SimpleNamespace(datastack_name='test',chunkedgraph=SimpleNamespace(
        get_roots=lambda ids,timestamp:np.array([{1:20,2:20,3:30,4:30,5:40}[int(a)] for a in ids],np.uint64)))
    ref=d/'snapshot-frozen.json'
    prepare(d/'human-frozen.json',ref,client,'2026-09-28T00:00:00+00:00')
    (src/'index.json').write_text(json.dumps([{'case_id':'case','snapshot_reference':file_hash(ref)}]))
    meta={'schema':'connectome-bundle-v1','cases':1,'files':{str(p.relative_to(src)):file_hash(p) for p in src.rglob('*') if p.is_file() and p.name!='bundle.json'}}
    (src/'bundle.json').write_text(json.dumps(meta))
    original=file_hash(src/'bundle.json')
    dest=tmp_path/'upgraded'
    assert upgrade_reference(src,dest,client)['cases']==1
    assert verify(src)['status']=='verified' and file_hash(src/'bundle.json')==original
    spec=json.loads((dest/'case/evaluator/snapshot-frozen.json').read_text())
    assert spec['scope_kind']=='full-cube'
    data=read(dest/'case/evaluator'/spec['counts'])
    assert sum(data['count'])==6 and int(data['unknown_voxels'])==0
    assert 5 in data['cell_atoms']
    with pytest.raises(FileExistsError):upgrade_reference(src,dest,client)
