"""Export saved observations and outward explanations as shareable H.264 movies."""
import argparse
from functools import lru_cache
import json
import math
from pathlib import Path
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .snapshots import snapshot

WIDTH, HEIGHT = 1080, 900


@lru_cache(maxsize=16)
def font(size):
    from matplotlib import get_data_path
    return ImageFont.truetype(str(Path(get_data_path()) / 'fonts/ttf/DejaVuSans.ttf'), size)


def wrapped(text, face, width=984):
    draw = ImageDraw.Draw(Image.new('RGB', (1,1)))
    result = []
    for paragraph in text.splitlines():
        line = ''
        for word in paragraph.split():
            # Split an exceptionally long coordinate/string instead of clipping it.
            pieces = []
            while draw.textlength(word, font=face) > width:
                cut = len(word)-1
                while cut > 1 and draw.textlength(word[:cut], font=face) > width:cut-=1
                pieces.append(word[:cut]);word=word[cut:]
            pieces.append(word)
            for piece in pieces:
                candidate = (line+' '+piece).strip()
                if draw.textlength(candidate, font=face)>width and line:
                    result.append(line);line=piece
                else:line=candidate
        result.append(line)
    return result or ['No outward explanation recorded.']


def reveal_lines(lines, characters):
    """Stable line breaks: revealing a partial word never reflows earlier text."""
    visible=[]
    for line in lines:
        visible.append(line[:max(0,characters)])
        characters-=len(line)+1
    return visible


@lru_cache(maxsize=32)
def image_panel(root, path):
    if not re.fullmatch(r'frames/[A-Za-z0-9_.-]+\.png',path):raise ValueError('Unsafe frame path')
    target=(Path(root)/path).resolve()
    if not target.is_relative_to((Path(root)/'frames').resolve()):raise ValueError('Frame outside image directory')
    with Image.open(target) as source:
        image=source.convert('RGB');image.thumbnail((492,440),Image.Resampling.LANCZOS)
        return image


def background(frame, view, root, title, model, number, total, variant, variants, source_label=""):
    canvas=Image.new('RGB',(WIDTH,HEIGHT),'#0c1922');draw=ImageDraw.Draw(canvas)
    draw.text((40,24),title,fill='#e4f3f0',font=font(32))
    draw.text((40,69),model,fill='#9ebac4',font=font(22))
    label='Starting view' if number==0 else f'{number} / {total} · '+frame['tool'].replace('_',' ')
    draw.text((40,112),label,fill='#83d8bd',font=font(23))
    if variants>1:draw.text((815,112),f'View {variant+1} / {variants}',fill='#9ebac4',font=font(21))
    # A fixed top explanation panel avoids moving either image as text grows.
    draw.line((40,300,1040,300),fill='#29434b',width=1)
    for key,x,label in [('em',32,'EM view'),('segment',552,'Segment view')]:
        draw.text((x+10,319),label,fill='#c4dce5',font=font(23))
        draw.rounded_rectangle((x,362,x+496,816),radius=8,fill='#09151d',outline='#243b45',width=2)
        if view.get(key):
            image=image_panel(str(root),view[key]);canvas.paste(image,(x+(496-image.width)//2,369+(440-image.height)//2))
        else:draw.text((x+20,568),'No recorded view',fill='#91adb9',font=font(22))
    if source_label:draw.text((40,848),source_label,fill='#7998a4',font=font(19))
    draw.rectangle((0,HEIGHT-5,int(WIDTH*(number+1)/(total+1)),HEIGHT-1),fill='#73d4b3')
    return canvas


def overlay(base, lines, characters=None, spinner=None, pulse=0, changed=()):
    canvas=base.copy();draw=ImageDraw.Draw(canvas)
    shown=lines if characters is None else reveal_lines(lines,characters)
    for i,line in enumerate(shown):draw.text((40,160+i*31),line,fill='#edf4f1',font=font(26))
    if spinner is not None:
        last = max((i for i, line in enumerate(shown) if line.strip()), default=0)
        x = 40 + draw.textlength(shown[last].rstrip(), font=font(26)) + 10
        y = 166 + last * 31
        if x + 22 > WIDTH - 32:
            x, y = 40, y + 31
        draw.arc((x,y,x+22,y+22),spinner*360,spinner*360+255,fill='#8ddac4',width=3)
    if pulse>0:
        quiet=np.array([36,59,69]);bright=np.array([115,224,184])
        color=tuple((quiet+(bright-quiet)*pulse).astype(int))
        for key,x in [('em',32),('segment',552)]:
            if key in changed:draw.rounded_rectangle((x,362,x+496,816),radius=8,outline=color,width=3)
    return canvas


def export_movie(data, images, output, title, model, seconds=6, fps=24, chars_per_second=55, source_label=""):
    if seconds<=0 or fps<1 or chars_per_second<=0:raise ValueError('Positive timing values required')
    import imageio_ffmpeg
    images=Path(images).resolve();output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    timeline=[];tick_total=0
    writer=imageio_ffmpeg.write_frames(str(output),(WIDTH,HEIGHT),fps=fps,codec='libx264',
        pix_fmt_out='yuv420p',quality=7,macro_block_size=2,
        output_params=['-movflags','+faststart','-preset','veryfast','-threads','2'])
    writer.send(None)
    try:
        frames=data['frames']
        for n,frame in enumerate(frames):
            text=frame.get('text') or 'No outward explanation recorded.'
            if frame.get('error'):text+='\nTool error: '+str(frame['error'])
            lines=wrapped(text,font(26),width=952);pages=[lines[i:i+4] for i in range(0,len(lines),4)]
            previous=frames[n-1] if n else frame
            before=(previous.get('views') or [previous])[-1]
            views=frame.get('views') or [frame]
            stages=[]
            if not n:
                stages=[('initial',views[-1],page,pi,0,1,max(1,round(min(seconds,3)*fps)),()) for pi,page in enumerate(pages)]
            else:
                for pi,page in enumerate(pages):
                    typing_seconds=len('\n'.join(page))/chars_per_second
                    ticks=max(1,math.ceil(typing_seconds*fps))
                    stages.append(('explanation',before,page,pi,0,1,ticks,()))
                # Keep the final explanation on screen while the tool is applied.
                stages.append(('loading',before,pages[-1],len(pages)-1,0,1,max(1,round(.45*fps)),()))
                prior=before
                for vi,view in enumerate(views):
                    changed=tuple(k for k in ('em','segment') if view.get(k)!=prior.get(k))
                    stages.append(('observation',view,pages[-1],len(pages)-1,vi,len(views),max(1,round(2*fps)),changed))
                    prior=view
            for phase,view,page,pi,vi,nviews,ticks,changed in stages:
                base=background(frame,view,images,title,model,n,len(frames)-1,vi,nviews,source_label)
                characters=len('\n'.join(page));last_key=None;pixels=None
                for tick in range(ticks):
                    shown=min(characters,int((tick+1)*chars_per_second/fps)) if phase=='explanation' else None
                    spinner=(tick/fps/.7)%1 if phase=='loading' else None
                    pulse=max(0,1-tick/(fps*.8)) if phase=='observation' and changed else 0
                    key=(shown,spinner,pulse)
                    if key!=last_key:
                        pixels=np.asarray(overlay(base,page,shown,spinner,pulse,changed));last_key=key
                    writer.send(pixels)
                timeline.append({'start_seconds':tick_total/fps,'duration_seconds':ticks/fps,'action':n,
                    'phase':phase,'tool':frame['tool'],'view':vi+1,'page':pi+1,
                    'explanation':'\n'.join(page),'em':view.get('em'),'segment':view.get('segment'),
                    'changed_views':list(changed)})
                tick_total+=ticks
    finally:writer.close()
    metadata={'title':title,'model':model,'source_label':source_label,'actions':len(data['frames'])-1,'duration_seconds':tick_total/fps,
              'fps':fps,'frames':tick_total,'chars_per_second':chars_per_second,
              'slides':len(timeline),'presentation':'clean-labels-v6',
              'private_reasoning_included':False,'timeline':timeline}
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    def stamp(t):
        ms=round(t*1000);return f'{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02},{ms%1000:03}'
    output.with_suffix('.srt').write_text('\n\n'.join(f"{i+1}\n{stamp(t['start_seconds'])} --> {stamp(t['start_seconds']+t['duration_seconds'])}\nAction {t['action']} · {t['tool']}\n{t['explanation']}" for i,t in enumerate(timeline))+'\n')
    return metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--session',type=Path)
    source.add_argument('--snapshot',type=Path)
    p.add_argument('--images',type=Path,help='Episode directory for a prepared snapshot')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--title',default='Connectome proofreading')
    p.add_argument('--model',required=True)
    p.add_argument('--source-label',default='',help='Optional short dataset attribution in the footer')
    p.add_argument('--seconds',type=float,default=6,help='Initial observation hold (up to 3 seconds); explanations follow character speed')
    p.add_argument('--fps',type=int,default=24)
    p.add_argument('--chars-per-second',type=float,default=55)
    a=p.parse_args()
    if a.session:data,_=snapshot(a.session);images=a.session
    else:
        if not a.images:p.error('--snapshot requires --images')
        data=json.loads(a.snapshot.read_text());images=a.images
    result=export_movie(data,images,a.out,a.title,a.model,a.seconds,a.fps,a.chars_per_second,a.source_label)
    print(json.dumps({k:v for k,v in result.items() if k!='timeline'}))


if __name__=='__main__':main()
