import numpy as np
from PIL import Image
from proofreading import Dataset,Session
from proofreading.volume import Grid
from proofreading.view import color
from proofreading.tools import schemas


def test_visible_listing_and_selection_colors(tmp_path):
    a,b=2**56+1,2**56+2
    atoms=np.zeros((32,32,2),dtype=np.uint64);atoms[:16]=1;atoms[16:]=2
    d=Dataset(Grid(atoms,[1,1,1],[0,0,0]),Grid(np.full(atoms.shape,100,np.uint8),[1,1,1],[0,0,0]),{1:a,2:b})
    s=Session(d,tmp_path/'episode')
    s.set_view(center_nm=[16,16,.5],width=32,height=32,pixel_size_nm=1,overlay=1)
    def image():
        r=s.look();return np.asarray(Image.open(s.output/r['image'])),r
    im,r=image();assert set(r['visible_segments'])=={str(a),str(b)}
    np.testing.assert_array_equal(im[16,4],color(a).astype(np.uint8))
    np.testing.assert_array_equal(im[16,25],color(b).astype(np.uint8))
    s.set_view(isolate=True)  # Legacy isolate flag cannot defeat automatic coloring.
    assert s.list_segments()['total']==2
    s.select([str(a)])
    im,r=image();assert r['visible_segments']==[str(a)]
    np.testing.assert_array_equal(im[16,4],color(a).astype(np.uint8))
    assert tuple(im[16,25])==(100,100,100)
    listed=s.list_segments();assert listed['total']==1
    assert listed['segments']==[{'id':str(a),'pixels':512,'selected':True,'color_rgb':color(a).round().astype(int).tolist()}]
    s.select([str(b)]);assert s.list_segments()['total']==2
    s.select([str(b)],mode='hide');assert s.list_segments()['total']==1
    s.select([],mode='replace');assert s.list_segments()['total']==1
    s.select([str(b)],mode='show');assert s.list_segments()['total']==2
    s.set_view(center_nm=[1000,1000,.5]);assert s.list_segments()['total']==0
    s.set_view(center_nm=[16,16,.5],overlay=0);assert s.list_segments()['total']==0
    assert 'inspect_segment' not in [x['name'] for x in schemas()]
    assert not s.execute('inspect_segment',{'segment':str(a)})['ok']

    s.set_view(overlay=1)
    preview=s.preview_merge([str(a),str(b)])
    preview_pixels=np.asarray(Image.open(s.output/preview['image']))
    s.commit(preview['preview_id'])
    committed,_=image()
    np.testing.assert_array_equal(preview_pixels,committed)
