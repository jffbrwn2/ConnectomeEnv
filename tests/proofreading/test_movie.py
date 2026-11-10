import json
from PIL import Image
import imageio_ffmpeg
import pytest
from proofreading.movie import export_movie, wrapped, font


def test_movie_encodes_complete_explanations_and_timeline(tmp_path):
    (tmp_path/'frames').mkdir()
    Image.new('RGB',(32,32),'cyan').save(tmp_path/'frames/em.png')
    Image.new('RGB',(32,32),'yellow').save(tmp_path/'frames/scene.png')
    text=' '.join('explanation'+str(i) for i in range(120))
    frame={'tool':'look','text':text,'em':'frames/em.png','segment':'frames/scene.png','em_step':0,'segment_step':0,'views':[]}
    movie=tmp_path/'movie.mp4'
    result=export_movie({'frames':[frame]},tmp_path,movie,'Synthetic test','Fake model',seconds=.2)
    assert result['slides']>1
    assert ' '.join(row['explanation'].replace('\n',' ') for row in result['timeline'])==text
    assert result['duration_seconds']==pytest.approx(result['slides']*round(.2*result['fps'])/result['fps'])
    reader=imageio_ffmpeg.read_frames(str(movie));metadata=next(reader);count=sum(1 for _ in reader)
    assert metadata['size']==(1080,900)
    assert count==result['frames']
    assert movie.with_suffix('.srt').is_file()
    with pytest.raises(FileExistsError):export_movie({'frames':[frame]},tmp_path,movie,'Test','Fake')


def test_long_token_is_wrapped_without_loss():
    text='x'*300
    assert ''.join(wrapped(text,font(26)))==text


def test_explanation_precedes_changed_images(tmp_path):
    (tmp_path/'frames').mkdir()
    for name,color in [('before','cyan'),('after','yellow')]:
        Image.new('RGB',(32,32),color).save(tmp_path/'frames'/f'{name}.png')
    before={'tool':'initial','text':'Starting view','em':'frames/before.png','segment':None,'em_step':0,'views':[]}
    after={'tool':'set_view','text':'I will inspect the next slice.','em':'frames/after.png','segment':None,'em_step':1,'views':[]}
    result=export_movie({'frames':[before,after]},tmp_path,tmp_path/'story.mp4','Test','Fake',seconds=.4)
    stages=result['timeline']
    assert [x['phase'] for x in stages]==['initial','explanation','loading','observation']
    assert stages[1]['em']=='frames/before.png'
    assert stages[2]['em']=='frames/before.png'
    assert stages[3]['em']=='frames/after.png'
    assert stages[3]['explanation']==stages[1]['explanation']
    assert stages[3]['changed_views']==['em']
    assert stages[1]['explanation']=='I will inspect the next slice.'
    import math
    expected=math.ceil(len(stages[1]['explanation'])/result['chars_per_second']*result['fps'])/result['fps']
    assert stages[1]['duration_seconds']==expected
    assert stages[2]['start_seconds']==pytest.approx(stages[1]['start_seconds']+expected)


def test_character_reveal_preserves_line_layout():
    from proofreading.movie import reveal_lines
    assert reveal_lines(['The neuron','continues here'],5)==['The n','']
    assert reveal_lines(['The neuron','continues here'],13)==['The neuron','co']
