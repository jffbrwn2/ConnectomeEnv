import numpy as np
import pytest
from proofreading.preparation import count_slabs, human_counts, native_counts, roots_at, InvalidCase
from proofreading.compact_scoring import pack, read
from proofreading.prepare_bundle import catalogue, choose_wave


def test_slab_statistics_equal_existing_voxel_packer(tmp_path):
    grid = np.array([0,1,1,2,3,3,3,4,4,4,4,5], dtype=np.uint64).reshape(2,2,3)
    atoms, counts = count_slabs(grid[:,:,i:i+1] for i in range(3))
    before = np.array([10,10,20,20,30],np.uint64)
    after = np.array([11,12,20,20,30],np.uint64)
    reference = np.zeros_like(grid)
    for a,r in zip(atoms,after): reference[grid==a] = r
    scope = np.isin(grid,atoms[before==10])
    path = tmp_path/'counts.npz'
    pack(grid,dict(zip(map(int,atoms),map(int,before))),reference,scope,path)
    actual=human_counts(atoms,counts,before,after,[10])
    for key,value in read(path).items(): np.testing.assert_array_equal(actual[key],value)


def test_native_download_uses_bounded_slabs_and_requested_mip():
    class Volume:
        info={'scales':[{'resolution':[4,4,40],'voxel_offset':[0,0,0],'size':[100,100,100]},
                        {'resolution':[8,8,40],'voxel_offset':[0,0,0],'size':[100,100,100]}]}
        def download(self,bbox,**kwargs):
            assert kwargs['mip']==1 and not kwargs['agglomerate']
            assert bbox.size3()[2]<=2
            np.testing.assert_array_equal(kwargs['coord_resolution'],[8,8,40])
            return np.ones(tuple(bbox.size3())+(1,),dtype=np.uint64)
    atoms,counts=native_counts(Volume(),np.array([8,8,40]),np.array([32,40,240]),slab_depth=2)
    assert atoms.tolist()==[1] and counts.tolist()==[60]


@pytest.mark.parametrize('values',[np.array([-1]),np.array([1.5]),np.array([0])])
def test_bad_segmentation_rejected(values):
    with pytest.raises(ValueError):count_slabs([values])


def row(key,op,root,x=0,kind='merge_edit'):
    return {'combined_sample_hash':key,'source_archive':'unified_mouse','sample_type':kind,
            'metadata':{'operation_id':op,'before_root_ids':[root],'after_root_ids':[root+100],
                        'interface_point_nm':[x,0,0]}}


def test_catalogue_uses_authoritative_operation_validation():
    rows=[row('a',1,10),row('b',2,20),row('c',3,30,100000)]
    geometry={'edge_nm':80000,'bbox_nm':{'min':[0,0,0]}}
    result=catalogue(rows,{'1','3'},geometry,{(1,0,0)})
    assert list(result)==['a'] and result['a']['cube_split']=='val'


def test_wave_preserves_base_and_releases_invalid_reservations():
    retained=row('a',1,10)
    conflicting=row('b',2,10,20000)
    failed=row('c',3,20,40000)
    replacement=row('d',4,20,40000)
    merge=row('e',5,30,80000,'split_edit')
    plan={'retained_rows':[retained],'candidates':[conflicting,failed,replacement,merge],'per_task':2}
    assert [r['combined_sample_hash'] for r in choose_wave(plan,[],['c'],8)]==['d','e']


def test_missing_historical_owner_is_ineligible():
    from types import SimpleNamespace
    client=SimpleNamespace(chunkedgraph=SimpleNamespace(get_roots=lambda *a,**k:np.array([0],np.uint64)))
    with pytest.raises(InvalidCase,match='ownership'):roots_at(client,np.array([1],np.uint64),None)


def test_preflight_archive_retains_audit_files(tmp_path):
    import tarfile
    from proofreading.prepare_bundle import archive_preflight
    path=tmp_path/'preflight'
    (path/'case').mkdir(parents=True)
    (path/'case/image.png').write_bytes(b'image-payload')
    (path/'status.json').write_text('{"status":"completed"}')
    result=archive_preflight(path)
    assert not path.exists() and result['files']==2
    with tarfile.open(result['path']) as archive:
        assert archive.extractfile('preflight/case/image.png').read()==b'image-payload'
