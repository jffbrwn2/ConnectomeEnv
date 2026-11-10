import pytest
from proofreading.extended_reference import relationship_changes, human_pair_changes


def audit(i,f,r):return relationship_changes(dict(enumerate(i)),dict(enumerate(f)),dict(enumerate(r)))

def test_other_end_correct_merge_with_benchmark_unresolved():
    d=audit([1,2,3],[4,2,4],[9,9,9])
    assert d['assessment']=='supported' and d['supported_joined_pairs']==1
    h=human_pair_changes({0:1,1:2,2:3},{0:4,1:2,2:4},{0:9,1:9,2:9},[1,2])
    assert h['human_target_joined_pairs']==0


def test_benchmark_plus_wrong_extra_is_mixed():
    d=audit([1,2,3],[4,4,4],[9,9,8])
    assert d['supported_joined_pairs']==1 and d['contradicted_joined_pairs']==2
    assert d['assessment']=='mixed'


def test_split_distinguishes_repair_from_fragmentation():
    d=audit([1,1,1],[2,3,4],[9,9,8])
    assert d['supported_separated_pairs']==2 and d['contradicted_separated_pairs']==1


def test_missing_reference_is_unresolved_not_wrong_or_correct():
    d=audit([1,2],[3,3],[9,0])
    assert d['assessment']=='unresolved' and d['unresolved_changed_pairs']==1
    assert d['contradicted_joined_pairs']==0


def test_renaming_or_undone_edit_is_no_change():
    assert audit([1,1,2],[4,4,5],[9,9,8])['assessment']=='no_net_relationship_change'


def test_preexisting_merge_error_not_charged_again():
    d=audit([1,1,2],[3,3,3],[9,8,9])
    assert d['supported_joined_pairs']==1 and d['contradicted_joined_pairs']==1


def test_reference_is_not_a_missing_atom_fallback():
    with pytest.raises(ValueError):relationship_changes({1:2},{1:2},{})


def test_relationship_counts_match_exhaustive_pairs():
    import random,itertools
    rng=random.Random(42)
    for trial in range(100):
        i=[rng.randrange(1,5) for _ in range(12)]
        f=[rng.randrange(1,5) for _ in i]
        r=[rng.randrange(4) for _ in i]
        got=audit(i,f,r)
        expected={k:0 for k in ('supported_joined_pairs','contradicted_joined_pairs','supported_separated_pairs','contradicted_separated_pairs','unresolved_changed_pairs')}
        for a,b in itertools.combinations(range(len(i)),2):
            old=i[a]==i[b];new=f[a]==f[b]
            if old==new:continue
            if not r[a] or not r[b]:expected['unresolved_changed_pairs']+=1;continue
            correct=new==(r[a]==r[b]);key=('supported_' if correct else 'contradicted_')+('joined_pairs' if new else 'separated_pairs')
            expected[key]+=1
        assert {k:got[k] for k in expected}==expected
        assert got['changed_pairs']==sum(expected.values())


def test_benchmark_merge_and_extra_join_are_separable():
    i={0:1,1:2,2:3};f={0:4,1:4,2:4};r={0:9,1:9,2:9}
    d=relationship_changes(i,f,r);h=human_pair_changes(i,f,r,[1,2])
    assert d['supported_joined_pairs']==3
    assert h['human_target_supported_joined_pairs']==1


def test_compressed_cells_match_all_supervoxels():
    from proofreading.extended_reference import relationship_changes_from_cells
    import random,collections
    rng=random.Random(321)
    for _ in range(100):
        cells=[(rng.randrange(1,4),rng.randrange(1,4),rng.randrange(4),rng.randrange(1,20)) for _ in range(20)]
        initial=[];final=[];refs=[]
        for i,f,r,n in cells:initial.extend([i]*n);final.extend([f]*n);refs.extend([r]*n)
        assert relationship_changes_from_cells(cells)==audit(initial,final,refs)


def test_project_complete_cells_matches_rule_evaluation():
    from proofreading.extended_reference import project_reference_cells,relationship_changes_from_cells
    from proofreading.streamed_evaluate import project_rules
    rules=[{'parents':['1','2'],'children':['10'],'assignments':None},{'parents':['10'],'children':['11','12'],'assignments':{'100':'11','101':'12','102':'11'}}]
    initial={100:1,101:1,102:2};ref={100:9,101:8,102:9}
    cells=project_reference_cells(1,[100,101],[9,8],rules)+project_reference_cells(2,[102],[9],rules)
    assert relationship_changes_from_cells(cells)==relationship_changes(initial,project_rules(initial,rules),ref)
    with pytest.raises(ValueError,match='omitted'):project_reference_cells(1,[103],[9],rules)


def test_merge_only_projection_and_large_integer_counts():
    from proofreading.extended_reference import project_reference_cells,relationship_changes_from_cells
    rules=[{'parents':['1','2'],'children':['9223372036854775809'],'assignments':None}]
    assert project_reference_cells(1,[100,101],[8,9],rules)==[(1,9223372036854775809,8,1),(1,9223372036854775809,9,1)]
    n=10**10
    result=relationship_changes_from_cells([(1,3,8,n),(2,3,8,n)])
    assert result['supported_joined_pairs']==n*n


def test_voxel_weighting_matches_dense_voxel_enumeration():
    from proofreading.extended_reference import voxel_weighted_relationship_changes
    import random
    rng=random.Random(74)
    for _ in range(100):
        atoms=range(8)
        i={a:rng.randrange(1,4) for a in atoms}
        f={a:rng.randrange(1,4) for a in atoms}
        r={a:rng.randrange(4) for a in atoms}
        w={a:rng.randrange(1,10) for a in atoms}
        vi=[];vf=[];vr=[]
        for a in atoms:vi.extend([i[a]]*w[a]);vf.extend([f[a]]*w[a]);vr.extend([r[a]]*w[a])
        dense=audit(vi,vf,vr);got=voxel_weighted_relationship_changes(i,f,r,w)
        for k in ('supported_joined_pairs','contradicted_joined_pairs','supported_separated_pairs','contradicted_separated_pairs','unresolved_changed_pairs','changed_pairs'):
            assert got[k.replace('_pairs','_voxel_pairs')]==dense[k]
        assert got['assessment']==dense['assessment']


def test_tiny_contaminant_has_small_weight_but_remains_mixed():
    from proofreading.extended_reference import voxel_weighted_relationship_changes
    d=voxel_weighted_relationship_changes({1:1,2:2,3:2},{1:3,2:3,3:3},{1:9,2:9,3:8},{1:1000,2:999,3:1})
    assert d['assessment']=='mixed'
    assert d['supported_fraction_of_known_changed_voxel_pairs']==pytest.approx(.999)


def test_voxel_counts_missing_or_estimated_rejected():
    from proofreading.extended_reference import voxel_weighted_relationship_changes
    for w in ({},{1:0},{1:-1},{1:1.5},{1:True}):
        with pytest.raises(ValueError):voxel_weighted_relationship_changes({1:1},{1:2},{1:9},w)


def test_voxel_pairs_do_not_overflow_uint64():
    import numpy as np
    from proofreading.extended_reference import voxel_weighted_relationship_changes
    n=10**12
    d=voxel_weighted_relationship_changes({1:1,2:2},{1:3,2:3},{1:9,2:9},{1:np.uint64(n),2:np.uint64(n)})
    assert d['supported_joined_voxel_pairs']==n*n
