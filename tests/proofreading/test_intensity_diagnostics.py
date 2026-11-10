import numpy as np
from proofreading.volume import Dataset,Grid
from proofreading.view import View,render


def test_intensity_diagnostics_exclude_outside_and_preserve_pixels():
    atoms=np.ones((2,2,2),dtype=np.uint64)
    d=Dataset(Grid(atoms,[1,1,1],[0,0,0]),Grid(np.full_like(atoms,100),[1,1,1],[0,0,0]))
    v=View([1,1,1],width=16,height=16,pixel_size_nm=1,contrast=[0,80])
    normal,_=render(d,d.initial,set(),set(),v)
    stats={};diagnostic,_=render(d,d.initial,set(),set(),v,diagnostics=stats)
    assert np.array_equal(normal,diagnostic)
    assert stats['valid_pixels']==4
    assert set(stats['percentiles'].values())=={100.0}
    assert stats['fraction_clipped_high']==1
    assert stats['fraction_clipped_low']==0
