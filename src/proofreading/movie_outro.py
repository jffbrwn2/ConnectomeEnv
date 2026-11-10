"""Presentation-only synchronized orbits of saved model and human partitions."""
import json
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .movie import WIDTH, HEIGHT, font
from .movie_intro import ease



def merge_additions(seed, rules):
    """Original roots added to the seed by a merge-only saved partition.

    Reject splits or unrelated edits: those require a different visualization,
    rather than silently presenting them as seed extensions.
    """
    seed = int(seed)
    current, roots = seed, {seed}
    for rule in rules:
        parents = set(map(int, rule['parents']))
        children = list(map(int, rule['children']))
        if rule['assignments'] is not None or len(children) != 1 or current not in parents:
            raise ValueError('Expected merges extending the starting fragment')
        roots.update(parents - {current})
        current = children[0]
    return roots - {seed}


def camera_at(t, start, bounds):
    lo, hi = np.asarray(bounds, dtype=float)
    end = (lo + hi) / 2
    # A bounding sphere fits every orientation; both panels use this same camera.
    height = max(float(np.linalg.norm(hi-lo)*1.15), float(start['view_height_nm'])*1.1, 1000.)
    move = ease((t-1)/4)
    turn = ease((t-5)/10)
    return {'target_nm': ((1-move)*np.asarray(start['target_nm'])+move*end).tolist(),
            'view_height_nm': math.exp((1-move)*math.log(start['view_height_nm'])+move*math.log(height)),
            'azimuth': float(start['azimuth'])+360*turn,
            'elevation': float(start['elevation'])}


def join_movies(parts, output):
    import imageio_ffmpeg
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory() as folder:
        listing = Path(folder)/'concat.txt'
        listing.write_text(''.join("file '"+str(Path(p).resolve()).replace("'", "'\\''")+"'\n" for p in parts))
        temporary = output.with_name(output.stem+'.partial.mp4')
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-y', '-f', 'concat',
                        '-safe', '0', '-i', str(listing), '-c', 'copy', '-movflags', '+faststart', str(temporary)], check=True)
        temporary.replace(output)


def export_outro(model_meshes, reference_meshes, start_camera, last_image, output,
                 title, model, note, provenance, renderer_factory=None):
    import imageio_ffmpeg
    from .raster import Renderer
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if not model_meshes or not reference_meshes:
        raise ValueError('Both comparison partitions must contain visible geometry')
    vertices = [v for _,v,_ in model_meshes+reference_meshes if len(v)]
    bounds = [np.min([v.min(0) for v in vertices],0), np.max([v.max(0) for v in vertices],0)]
    left, right = [(renderer_factory or Renderer)() for _ in range(2)]
    last_image = last_image.convert('RGB').resize((WIDTH,HEIGHT))
    fps, seconds = 24, 18
    writer = imageio_ffmpeg.write_frames(str(output),(WIDTH,HEIGHT),fps=fps,codec='libx264',
        pix_fmt_out='yuv420p',quality=7,macro_block_size=2,
        output_params=['-movflags','+faststart','-preset','veryfast','-threads','2'])
    writer.send(None)
    edit_view = provenance.get('presentation') == 'assembled-segment'
    held = None
    try:
        for n in range(fps*seconds):
            t=n/fps
            # The final three seconds hold still without repeatedly redrawing meshes.
            if held is None or t<=15:
                camera=camera_at(t,start_camera,bounds)
                canvas=Image.new('RGB',(WIDTH,HEIGHT),'#0c1922');draw=ImageDraw.Draw(canvas)
                draw.text((32,22),title,font=font(30),fill='#e4f3f0')
                draw.text((32,66),provenance.get('comparison_title', 'Assembled segment: model versus human' if edit_view else 'Final result compared with the recorded human correction'),font=font(22),fill='#9ebac4')
                draw.text((32,122),model,font=font(27),fill='#7fe0be')
                draw.text((556,122),provenance.get('reference_label', 'Human edit' if edit_view else 'Human reference'),font=font(27),fill='#7fe0be')
                for x,renderer,meshes in [(30,left,model_meshes),(554,right,reference_meshes)]:
                    scene=renderer.render(meshes,camera,bounds,np.empty((0,3)),None,show_bounds=False)['image']
                    # Preserve aspect ratio; never independently fit either panel.
                    scene.thumbnail((496,590),Image.Resampling.LANCZOS)
                    canvas.paste(scene,(x+(496-scene.width)//2,170+(590-scene.height)//2))
                    draw.rounded_rectangle((x,170,x+496,760),radius=8,outline='#35535e',width=2)
                draw.text((32,787),note,font=font(21),fill='#e4f3f0')
                draw.text((32,827),provenance.get('color_note', 'Same segment color, camera and scale.' if edit_view else 'Matched camera/scale. Gray: unscored neighbor context.'),font=font(20),fill='#9ebac4')
                draw.text((32,858),provenance.get('scope_note', 'Starting fragment and its joined material only.' if edit_view else 'Local human edit reference; outside-target material is unscored.'),font=font(19),fill='#9ebac4')
                held=canvas
            canvas=held
            if t<1:
                canvas=Image.blend(last_image,canvas,ease(t))
            writer.send(np.asarray(canvas))
            if n in (0,24,120,240,360,431):
                canvas.save(output.with_name(output.stem+f'-review-{n}.png'))
    finally:
        writer.close()
    result={'schema':1,'presentation':'synchronized-human-reference-outro-v1','seconds':seconds,
            'fps':fps,'frames':fps*seconds,'orbit_degrees':360,'shared_bounds_nm':np.asarray(bounds).tolist(),
            'end_camera':camera_at(15,start_camera,bounds),'reference_scope':'recorded local correction, not whole-neuron certification',
            'provenance':provenance}
    output.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
