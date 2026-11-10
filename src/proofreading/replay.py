"""Re-execute recorded actions and export the exact images the agent saw."""

import base64
import html
import json
import io
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .session import Session
from .harness import image_paths


def replay(dataset, source, output):
    source = Path(source)
    meta = json.loads((source / "session.json").read_text())
    if meta["dataset_sha256"] != dataset.fingerprint:
        raise ValueError("Replay dataset fingerprint mismatch")
    session = Session(dataset, output, seeded_split_policy=meta.get("seeded_split_policy", "connected_components"),
                      split_algorithm=meta.get("split_algorithm", "min_cut"))
    count = 0
    for line in (source / "events.jsonl").read_text().splitlines():
        event = json.loads(line)
        response = session.execute(event["tool"], event["arguments"], replay_legacy=event["tool"] in {"inspect_segment", "undo", "redo"} and event["response"]["ok"], refresh_selection=bool(list(image_paths(event["response"]))))
        if response["ok"] != event["response"]["ok"]:
            raise ValueError(f"Replay status mismatch at step {event['step']}")
        old = list(image_paths(event["response"]))
        new = list(image_paths(response))
        if not any(p.endswith("-scene.png") for p in old):
            new = [p for p in new if not p.endswith("-scene.png")]
        if len(old) != len(new):
            raise ValueError("Replay image count changed")
        for a, b in zip(old, new):
            if (source / a).read_bytes() != (session.output / b).read_bytes():
                raise ValueError(f"Replay image mismatch at step {event['step']}")
            count += 1
    with (
        np.load(source / "partition.npz") as expected,
        np.load(session.output / "partition.npz") as actual,
    ):
        if any(not np.array_equal(expected[k], actual[k]) for k in expected.files):
            raise ValueError("Replay final partition mismatch")
    result = {
        "status": "verified",
        "actions": len(session.events),
        "exact_images": count,
        "partition_equal": True,
    }
    (session.output / "replay-verification.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result



def observation_images(directory, response):
    """Keep a recorded EM/scene pair together; preserve legacy single images."""
    if isinstance(response, dict):
        if isinstance(response.get("image"), str):
            em = Image.open(directory / response["image"]).convert("RGB")
            scene = response.get("scene_observation")
            if scene:
                mesh = Image.open(directory / scene["image"]).convert("RGB")
                canvas = Image.new("RGB", (1800, 740), "#101722")
                draw = ImageDraw.Draw(canvas)
                for x, label, im in [(0, "3D segment view", mesh), (900, "EM view", em)]:
                    draw.text((x+20, 12), label, fill="white")
                    im.thumbnail((900, 700))
                    canvas.paste(im, (x+(900-im.width)//2, 40+(700-im.height)//2))
                yield canvas
            else:
                yield em
            return
        for value in response.values():
            yield from observation_images(directory, value)
    elif isinstance(response, list):
        for value in response:
            yield from observation_images(directory, value)

def export(session_dir):
    directory = Path(session_dir)
    events = [
        json.loads(s) for s in (directory / "events.jsonl").read_text().splitlines()
    ]
    harness = directory / "harness.json"
    identity = (
        ("Model: " + json.loads(harness.read_text())["model"])
        if harness.exists()
        else "Recorded tool session · not an autonomous model benchmark"
    )
    slides = []
    movie = []
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    for event in events:
        for image in observation_images(directory, event["response"]):
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            raw = buffer.getvalue()
            title = f"{event['step']:03d} / {event['tool'].replace('_', ' ')} · revision {event['revision']}"
            slides.append(
                {
                    "title": title,
                    "arguments": event["arguments"],
                    "image": "data:image/png;base64," + base64.b64encode(raw).decode(),
                }
            )
            frame = Image.new("RGB", (1100, 820), "#101722")
            draw = ImageDraw.Draw(frame)
            draw.text((28, 24), "CONNECTOME / " + title, fill="#dceafa", font=font)
            draw.text((28, 60), identity, fill="#95b5ca", font=font)
            image.thumbnail((1030, 675))
            frame.paste(image, ((1100 - image.width) // 2, 110))
            draw.text(
                (28, 785),
                "Exact recorded observations · no reference answers shown to the agent",
                fill="#91aaae",
                font=font,
            )
            movie.append(frame)
    if not slides:
        raise ValueError("Session contains no images")
    payload = json.dumps(slides).replace("<", "\\u003c")
    page = (
        """<!doctype html><meta charset="utf-8"><title>Connectome replay</title><style>body{background:#101722;color:#eaf3ff;font:16px system-ui;margin:25px auto;max-width:1100px}img{max-width:100%;height:70vh;object-fit:contain;display:block;margin:auto}button,input{margin:10px;padding:10px}pre{white-space:pre-wrap;color:#a2bdce}small{color:#a2bdce}</style><h1>Connectome / episode replay</h1><small>"""
        + html.escape(identity)
        + """</small><h2 id="title"></h2><img id="image"><button id="play">Play</button><input id="scrub" type="range" min="0" value="0"><span id="count"></span><pre id="args"></pre><script>const slides="""
        + payload
        + """;let i=0,timer=null;const $=id=>document.getElementById(id);$('scrub').max=slides.length-1;function show(){let s=slides[i];$('title').textContent=s.title;$('image').src=s.image;$('args').textContent=JSON.stringify(s.arguments,null,2);$('scrub').value=i;$('count').textContent=(i+1)+' / '+slides.length} $('scrub').oninput=()=>{i=Number($('scrub').value);show()};$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='Play'}else{timer=setInterval(()=>{i=(i+1)%slides.length;show()},1200);$('play').textContent='Pause'}};show();</script>"""
    )
    (directory / "replay.html").write_text(page)
    movie[0].save(
        directory / "replay.gif",
        save_all=True,
        append_images=movie[1:],
        duration=1200,
        loop=0,
    )
    import imageio_ffmpeg

    writer = imageio_ffmpeg.write_frames(
        str(directory / "replay.mp4"),
        (1100, 820),
        fps=5,
        codec="libx264",
        quality=8,
        macro_block_size=2,
    )
    writer.send(None)
    try:
        for frame in movie:
            for _ in range(6):
                writer.send(np.asarray(frame))
    finally:
        writer.close()
    return {
        "html": str(directory / "replay.html"),
        "gif": str(directory / "replay.gif"),
        "mp4": str(directory / "replay.mp4"),
        "frames": len(slides),
    }
