import pytest
from proofreading.streaming_partition import StreamingPartition


class Source:
    def __init__(self):
        self.labels={1:10,2:10,3:20,4:20,5:30}
        self.seen={1,3,5};self.complete_calls=[]
    def owner(self,a):return self.labels[a]
    def observed(self,r):return [a for a in self.seen if self.labels[a]==r]
    def members(self,r):
        self.complete_calls.append(r)
        return [a for a,s in self.labels.items() if s==r]


def test_merge_applies_to_unloaded_material_and_undo_snapshot():
    source=Source();p=StreamingPartition(source);before=p.copy()
    p.apply_groups('merge',[10,20],[{1,3}],[100])
    assert not source.complete_calls  # No whole-neuron membership query for a merge.
    source.seen.update([2,4])
    assert p[2]==p[4]==100 and p.atoms_for(100)=={1,2,3,4}
    assert before[2]==10 and before[4]==20
    assert not p.atoms_for(10) and not p.atoms_for(20)


def test_split_cannot_silently_omit_unloaded_atoms():
    p=StreamingPartition(Source())
    with pytest.raises(ValueError,match='complete parent'):
        p.apply_groups('split',[10],[{1},{999}],[100,101])
    assert p.rules==()
    p.apply_groups('split',[10],[{1},{2}],[100,101])
    assert p[1]==100 and p[2]==101


def test_split_merge_and_shared_history():
    source=Source();p=StreamingPartition(source)
    p.apply_groups('merge',[10,20],[{1,3}],[100]);merged=p.copy()
    p.apply_groups('split',[100],[{1,4},{2,3}],[101,102])
    assert merged.rules[0] is p.rules[0]
    assert [p[a] for a in [1,2,3,4]]==[101,102,102,101]
    assert merged[1]==merged[4]==100
    source.seen.update([2,4])
    p.apply_groups('merge',[101,30],[{1,4,5}],[103])
    assert p.atoms_for(103,complete=True)=={1,4,5}
    assert p.atoms_for(102,complete=True)=={2,3}


def test_source_membership_drift_fails_closed():
    source=Source();p=StreamingPartition(source)
    p.apply_groups('split',[10],[{1},{2}],[100,101])
    source.labels[6]=10
    with pytest.raises(ValueError,match='membership changed'):p[6]
