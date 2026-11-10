"""Reconstruct logged scene geometry without consulting evaluation labels.

Native rectangular geometry channels retain the recorded framing. The model
adapter letterboxes these into a square; it never widens the visible field.
The geometry renderer and pair fusion are the existing training implementations.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image

CHANNELS = ['silhouette', 'depth', 'nx', 'ny', 'nz', 'a_mask', 'b_mask']


def letterbox(channels, resolution=224):
    if resolution < 1:
        raise ValueError('Positive output resolution required')
    _, height, width = channels.shape
    scale = resolution / max(height, width)
    w, h = max(1, round(width * scale)), max(1, round(height * scale))
    x, y = (resolution-w)//2, (resolution-h)//2
    result = np.zeros((channels.shape[0], resolution, resolution), np.float16)
    for i, channel in enumerate(channels):
        result[i, y:y+h, x:x+w] = np.asarray(Image.fromarray(channel.astype(np.float32)).resize(
            (w, h), Image.Resampling.NEAREST), dtype=np.float16)
    return result, dict(x=x, y=y, width=w, height=h, resolution=resolution)


def render_channels(meshes, camera, size=(900, 700), resolution=224):
    """Render one or two exact mesh snapshots using training channel semantics.

    meshes: ordered (segment_id, vertices_nm, faces) tuples; order defines A/B.
    Returns native [7,H,W], model [1,7,R,R], and provenance. Pair masks are
    independent silhouettes, including occluded portions, as during training.
    Pair depth is normalized independently before training's pair fusion. This
    legacy convention is retained, not claimed to be physical joint depth.
    """
    from cloudvolume import Mesh
    from rendering.geometry_renderer import (
        create_geometry_viewer, _clear_scene, _apply_normal_material,
        _render_color_buffer, _read_depth_buffer,
    )
    from rendering.camera_utils import configure_camera_for_view
    from rendering.render_utils import CameraViewSpec
    from foundation.scenes.fused_merge_pair_scene import fuse_merge_pair_views
    from .scene import validate
    validate(camera)
    if len(meshes) not in (1, 2) or len({str(m[0]) for m in meshes}) != len(meshes):
        raise ValueError('Exactly one or two distinct segments required; never silently discard other visible segments')
    width, height = map(int, size)
    if width <= 0 or height <= 0:
        raise ValueError('Positive image dimensions required')
    target = np.asarray(camera['target_nm'], float)
    az, el = np.deg2rad([camera['azimuth'], camera['elevation']])
    eye = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
    right = np.array([-np.sin(az), np.cos(az), 0.])
    up = np.cross(eye, right)
    relative = []
    hashes = []
    reach = 1.
    for sid, vertices, faces in meshes:
        vertices, faces = np.asarray(vertices), np.asarray(faces)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError('Expected triangular meshes with Nx3 vertices and Mx3 faces')
        if not len(vertices) or not len(faces) or not np.isfinite(vertices).all():
            raise ValueError('Empty or nonfinite mesh')
        if faces.dtype.kind not in 'iu' or faces.min() < 0 or faces.max() >= len(vertices):
            raise ValueError('Invalid face indices')
        v = (vertices.astype(np.float64)-target).astype(np.float32)
        reach = max(reach, float(np.max(np.abs(v @ eye))))
        relative.append((v, faces))
        digest = hashlib.sha256()
        digest.update(np.asarray(vertices, dtype='<f8').tobytes())
        digest.update(np.asarray(faces, dtype='<u8').tobytes())
        hashes.append(dict(segment_id=str(sid), sha256=digest.hexdigest()))
    distance = reach + camera['view_height_nm'] * 2 + 1
    viewer = create_geometry_viewer()
    parts = []
    try:
        for vertices, faces in relative:
            _clear_scene(viewer)
            viewer.add(Mesh(vertices, faces), color='white')
            _apply_normal_material(viewer)
            spec = CameraViewSpec(name='recorded', position_nm=tuple(eye*distance),
                target_nm=(0.,0.,0.), up=tuple(up),
                ortho_extent_nm=camera['view_height_nm']/2)
            configure_camera_for_view(viewer, spec)
            viewer.camera.depth_range = (1., distance+reach+1.)
            viewer.camera.width = camera['view_height_nm'] * width / height
            viewer.camera.height = camera['view_height_nm']
            rgba = _render_color_buffer(viewer, (width, height))
            depth = _read_depth_buffer(viewer)
            if rgba.shape[:2] != (height,width) or depth.shape != (height,width):
                raise ValueError('Renderer changed requested image dimensions')
            foreground = depth < 1.
            normalized = np.zeros_like(depth)
            if foreground.any():
                values = depth[foreground]
                normalized[foreground] = (values-values.min())/(values.max()-values.min()+1e-8)
            normals = rgba[:,:,:3].astype(np.float32)/255*2-1
            normals[~foreground] = 0
            parts.append(np.concatenate([foreground[None], normalized[None], normals.transpose(2,0,1)]).astype(np.float16))
    finally:
        viewer.close()
    if len(parts) == 1:
        native = np.concatenate([parts[0], parts[0][:1], np.zeros_like(parts[0][:1])])
    else:
        pair = np.concatenate([*parts, np.zeros_like(parts[0][:1])])[None]
        native = fuse_merge_pair_views(pair)[0]
    model, padding = letterbox(native, resolution)
    from importlib.metadata import version
    metadata = dict(schema=1, channels=CHANNELS, camera=camera, image_size=[width,height],
        renderer_versions={name: version(name) for name in ('pygfx','wgpu','octarine3d')},
        meshes=hashes, padding=padding, normal_encoding='training MeshNormalMaterial',
        depth_encoding='per-segment foreground min/max; training pair fusion',
        model_view_distribution='recorded LLM camera, not canonical training views',
        masks='independent silhouettes including occluded portions',
        reference_labels_used=False)
    return native, model[None], metadata


def write_reconstruction(meshes, camera, output, *, size=(900,700), resolution=224, provenance=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    native, model, metadata = render_channels(meshes, camera, size, resolution)
    metadata['provenance'] = provenance or {}
    np.savez_compressed(output/'geometry.npz', native=native, views=model)
    Image.fromarray((native[0]*255).astype(np.uint8)).save(output/'silhouette.png')
    Image.fromarray((model[0,0]*255).astype(np.uint8)).save(output/'model-silhouette.png')
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


def reconstruct_observation(dataset, episode, image, output, *, resolution=224, export_channels=True):
    """Replay to an exact saved scene image and capture its historical meshes.

    Fail closed on dataset, action status, revision or RGB pixel mismatch.
    Capturing at render_scene also handles temporary preview partitions and
    undo correctly. Run in a dedicated process: the capture hook is global.
    No model calls or reference-label access occur. Existing episodes untouched.
    """
    import copy
    from unittest.mock import patch
    from .session import Session
    from . import scene
    from .harness import image_paths
    episode, output = Path(episode), Path(output)
    source_image = (episode/image).resolve()
    source_image.relative_to((episode/'frames').resolve())
    if not source_image.is_file():
        raise ValueError('Recorded scene image is missing')
    meta = json.loads((episode/'session.json').read_text())
    if meta['dataset_sha256'] != dataset.fingerprint:
        raise ValueError('Dataset fingerprint mismatch')
    events = [json.loads(s) for s in (episode/'events.jsonl').read_text().splitlines() if s.strip()]
    matching = [e for e in events if image in list(image_paths(e['response']))]
    if len(matching) != 1:
        # Repeated references within one event are harmless; event matching is unique.
        raise ValueError('Image must identify exactly one recorded event')
    target_event = matching[0]
    output.mkdir(parents=True, exist_ok=False)
    session = Session(dataset, output/'replay',
        seeded_split_policy=meta.get('seeded_split_policy','connected_components'),
        split_algorithm=meta.get('split_algorithm','min_cut'))
    captured = []
    original = scene.render_scene
    def capture(state, path):
        result = original(state, path)
        if Path(path).relative_to(session.output).as_posix() == image:
            meshes = [(sid, v.copy(), f.copy()) for sid,v,f in scene.geometry(state)]
            captured.append((meshes, copy.deepcopy(state.scene), state.revision))
        return result
    with patch.object(scene, 'render_scene', capture):
        for event in events:
            response = session.execute(event['tool'], event['arguments'],
                replay_legacy=event['tool'] in {'inspect_segment','undo','redo'} and event['response']['ok'],
                refresh_selection=bool(list(image_paths(event['response']))))
            if response['ok'] != event['response']['ok'] or session.serial != event['step']:
                raise ValueError(f"Replay diverged at step {event['step']}")
            if event['step'] == target_event['step']:
                break
    if len(captured) != 1:
        raise ValueError('Requested scene was not reconstructed exactly once')
    saved = np.asarray(Image.open(source_image).convert('RGB'))
    actual = np.asarray(Image.open(session.output/image).convert('RGB'))
    if saved.shape != actual.shape or not np.array_equal(saved, actual):
        raise ValueError('Replayed RGB pixels differ; refusing unverified geometry export')
    meshes, camera, revision = captured[0]
    # Persist input geometry, so future tensor generation needs neither CAVE nor replay.
    arrays = {}
    for i,(sid,v,f) in enumerate(meshes):
        arrays.update({f'vertices_{i}': v, f'faces_{i}': f})
    np.savez_compressed(output/'mesh-snapshot.npz', **arrays)
    provenance = dict(source_episode=str(episode), source_image=image,
        image_sha256=hashlib.sha256(source_image.read_bytes()).hexdigest(),
        dataset_sha256=dataset.fingerprint, step=target_event['step'], revision=revision,
        rgb_pixels_equal=True, image_size=[saved.shape[1],saved.shape[0]], segment_ids=[str(sid) for sid,_,_ in meshes])
    (output/'snapshot.json').write_text(json.dumps(dict(camera=camera, **provenance),indent=2)+'\n')
    if not export_channels:
        return dict(status='verified_mesh_snapshot', camera=camera, **provenance)
    return write_reconstruction(meshes,camera,output/'channels',
        size=(saved.shape[1],saved.shape[0]),resolution=resolution,provenance=provenance)


def reconstruct_snapshot(snapshot, output, *, resolution=224):
    """Render a previously verified mesh snapshot without CAVE or tool replay."""
    snapshot = Path(snapshot)
    provenance = json.loads((snapshot/'snapshot.json').read_text())
    if provenance.get('rgb_pixels_equal') is not True:
        raise ValueError('Snapshot lacks exact RGB replay verification')
    with np.load(snapshot/'mesh-snapshot.npz', allow_pickle=False) as data:
        meshes = [(int(sid), data[f'vertices_{i}'], data[f'faces_{i}'])
                  for i,sid in enumerate(provenance['segment_ids'])]
    return write_reconstruction(meshes, provenance['camera'], output,
        size=tuple(provenance.get('image_size', (900,700))),
        resolution=resolution, provenance=provenance)
