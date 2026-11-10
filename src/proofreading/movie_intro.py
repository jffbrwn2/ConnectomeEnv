"""A contextual mesh orbit and camera move into an unchanged recorded movie."""
import argparse,json,math,subprocess,tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from .movie import WIDTH,HEIGHT,font,wrapped,background,overlay
from .raster import Renderer
from .view import View


def ease(t):
    t=float(np.clip(t,0,1));return t*t*t*(t*(t*6-15)+10)


def export_intro(mesh_file, observation_file, snapshot_file, images, replay, output, title, model, source_label):
    import imageio_ffmpeg
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with np.load(mesh_file,allow_pickle=False) as data:
        meshes=[(int(data['id0']),data['v0'].copy(),data['f0'].copy())]
    observation=json.loads(Path(observation_file).read_text())['observation']
    camera=observation['scene_observation']['scene']
    transform=observation['transform'];fields=View.__dataclass_fields__
    view=View(**{k:v for k,v in transform.items() if k in fields})
    plane=[view.xyz(x,y) for x,y in [(-.5,-.5),(view.width-.5,-.5),(view.width-.5,view.height-.5),(-.5,view.height-.5),(-.5,-.5)]]
    vertices=meshes[0][1];low,high=vertices.min(0),vertices.max(0)
    center=(low+high)/2;overview_height=float(np.linalg.norm(high-low)*1.2)
    data=json.loads(Path(snapshot_file).read_text());first=data['frames'][0]
    initial_view=(first.get('views') or [first])[-1]
    base=background(first,initial_view,Path(images),title,model,0,len(data['frames'])-1,0,1,source_label)
    final=overlay(base,wrapped(first['text'],font(26),952)[:4])
    # Final image rectangle exactly matches the recorded movie's segment panel.
    with Image.open(Path(images)/initial_view['segment']) as im:
        im.thumbnail((492,440),Image.Resampling.LANCZOS);fw,fh=im.size
    end=np.array([552+(496-fw)//2,369+(440-fh)//2,fw,fh],float)
    start=np.array([110,144,860,669],float)
    fps=24;duration=15;count=fps*duration;renderer=Renderer()
    with tempfile.TemporaryDirectory(prefix='connectome-intro-') as tmp:
        tmp=Path(tmp);intro=tmp/'intro.mp4'
        writer=imageio_ffmpeg.write_frames(str(intro),(WIDTH,HEIGHT),fps=fps,codec='libx264',pix_fmt_out='yuv420p',quality=7,macro_block_size=2,
            output_params=['-movflags','+faststart','-preset','veryfast','-threads','2']);writer.send(None)
        try:
            for frame in range(count):
                t=frame/fps
                turn=ease((t-.75)/8)
                move=ease((t-8.75)/5.25)
                az=camera['azimuth']+360*turn
                current={'target_nm':((1-move)*center+move*np.asarray(camera['target_nm'])).tolist(),
                         'azimuth':az,'elevation':camera['elevation']+8*math.sin(2*math.pi*turn)*(1-move),
                         'view_height_nm':math.exp((1-move)*math.log(overview_height)+move*math.log(camera['view_height_nm']))}
                scene=renderer.render(meshes,current,[low,high],plane,None,show_bounds=False)['image']
                # Reveal the paired observation layout as the camera approaches it.
                blank=Image.new('RGB',(WIDTH,HEIGHT),'#0c1922')
                blank.paste(base.crop((0,0,WIDTH,140)),(0,0));blank.paste(base.crop((0,824,WIDTH,HEIGHT)),(0,824))
                canvas=Image.blend(blank,base,move)
                text_alpha=ease((move-.75)/.25)
                text_box=(32,144,1048,295)
                canvas.paste(Image.blend(canvas.crop(text_box),final.crop(text_box),text_alpha),text_box[:2])
                x,y,w,h=np.rint(start*(1-move)+end*move).astype(int)
                scene=scene.resize((w,h),Image.Resampling.LANCZOS)
                canvas.paste(scene,(x,y))
                # Historical full mesh and crop voxel surface differ in tessellation.
                # Dissolve into the exact recorded pixels before replay begins.
                canvas=Image.blend(canvas,final,ease((t-14)/(.95)))
                writer.send(np.asarray(canvas))
                if frame in (0,108,216,300,359):canvas.save(output.with_name(output.stem+f'-review-{frame}.png'))
        finally:writer.close()
        playlist=tmp/'concat.txt'
        def quoted(p):return "file '"+str(Path(p).resolve()).replace("'","'\\''")+"'\n"
        playlist.write_text(quoted(intro)+quoted(replay))
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-f','concat','-safe','0','-i',str(playlist),'-c','copy','-movflags','+faststart',str(output)],check=True)
    original=json.loads(Path(replay).with_suffix('.json').read_text())
    metadata={**original,'presentation':'historical-orbit-intro-v1','intro_seconds':duration,
              'intro_provenance':{'root_id':str(meshes[0][0]),'source':'historical root mesh; contextual camera animation, not model actions',
                                  'mesh_bounds_nm':[low.tolist(),high.tolist()]},
              'slides':len(original['timeline'])+1,'frames':original['frames']+count,'duration_seconds':original['duration_seconds']+duration,
              'timeline':[{'phase':'overview','start_seconds':0,'duration_seconds':duration}]+[{**r,'start_seconds':r['start_seconds']+duration} for r in original['timeline']]}
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('mesh','observation','snapshot','images','replay','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--title',required=True);p.add_argument('--model',required=True);p.add_argument('--source-label',default='')
    a=p.parse_args();r=export_intro(a.mesh,a.observation,a.snapshot,a.images,a.replay,a.out,a.title,a.model,a.source_label)
    print(json.dumps({k:v for k,v in r.items() if k!='timeline'}))


if __name__=='__main__':main()
