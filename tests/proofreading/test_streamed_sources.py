import numpy as np
from proofreading.streamed_sources import StreamedSources
from proofreading.view import View,visible_segments
from proofreading.streaming_partition import StreamingPartition


class Owners:
    def __init__(self):self.known={}
    def ensure(self,atoms):self.known.update({int(a):int(a)+100 for a in atoms})
    def owner(self,a):return self.known[a]
    def observed(self,r):return [a for a,s in self.known.items() if s==r]


def test_exploring_beyond_initial_view_discovers_clickable_ids_and_matching_lod():
    scales=[{'resolution':[n,n,40],'voxel_offset':[0,0,0],'size':[4096//n,4096//n,64]} for n in (8,16,32)]
    config={k:{'source':k,'info':{'data_type':'uint8' if k=='em' else 'uint64','scales':scales}} for k in ('em','segmentation')}
    calls=[];owners=Owners()
    def load(channel,mip,lo,hi):
        calls.append((channel,mip,tuple(lo)))
        res=np.asarray(scales[mip]['resolution'])
        x=(np.arange(lo[0],hi[0])+.5)*res[0]
        labels=np.where(x<2048,1,2).astype(np.uint64)
        return np.broadcast_to(labels[:,None,None],hi-lo).astype(np.uint8 if channel=='em' else np.uint64).copy()
    dataset=StreamedSources(config,load,owners)
    partition=StreamingPartition(owners)
    first=View([1024,1024,1000],width=32,height=32,pixel_size_nm=16)
    _,_,visible,_=visible_segments(dataset,partition,set(),set(),first)
    assert visible==[101]
    second=View([3072,1024,1000],width=32,height=32,pixel_size_nm=32)
    _,_,visible,_=visible_segments(dataset,partition,set(),set(),second)
    assert visible==[102]
    em,seg=dataset.sampling_grids(second)
    atom,valid=seg.nearest(second.xyz(16,16))
    assert valid and partition[int(atom)]==102
    assert np.array_equal(em.resolution,seg.resolution) and em.resolution[0]==32
    count=len(calls);seg.nearest(second.xyz(17,16));assert len(calls)==count
    second.em_resolution_nm=8
    assert dataset.sampling_grids(second)[0].resolution[0]==8
