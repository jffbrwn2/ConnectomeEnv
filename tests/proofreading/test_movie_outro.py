import numpy as np
import pytest
from proofreading.movie_outro import camera_at


def test_orbit_uses_shared_extent_and_finishes_one_turn():
    start={'target_nm':[2,3,4],'view_height_nm':100,'azimuth':35,'elevation':25}
    bounds=[[-1000,-2000,-3000],[1000,2000,3000]]
    first=camera_at(0,start,bounds);end=camera_at(15,start,bounds)
    assert first['target_nm']==start['target_nm']
    assert first['view_height_nm']==pytest.approx(100)
    assert end['target_nm']==[0,0,0]
    assert end['azimuth']==395
    assert end['elevation']==start['elevation']
    assert end['view_height_nm']>np.linalg.norm(np.subtract(bounds[1],bounds[0]))
    assert camera_at(18,start,bounds)==end


def test_zoom_never_finishes_smaller_than_initial_view():
    start={'target_nm':[0,0,0],'view_height_nm':50000,'azimuth':0,'elevation':0}
    assert camera_at(15,start,[[-1,-1,-1],[1,1,1]])['view_height_nm']==pytest.approx(55000)


def test_merge_additions_excludes_start_and_resolves_chained_edits():
    from proofreading.movie_outro import merge_additions
    rules = [dict(parents=['10', '20'], children=['100'], assignments=None),
             dict(parents=['100', '30'], children=['101'], assignments=None)]
    assert merge_additions('10', rules) == {20, 30}
    assert merge_additions('10', []) == set()
    with pytest.raises(ValueError):
        merge_additions('99', rules)
    with pytest.raises(ValueError):
        merge_additions('10', [dict(parents=['10'], children=['100', '101'], assignments={'1': '100'})])
