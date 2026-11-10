"""Persistent physical 3D camera, exact selected geometry, and browser GLB export."""
import json
import struct
import numpy as np
from .view import color
from .volume import vector


def geometry(session, ids=None, stride=1):
    """Bounded per-session mesh cache; camera movement never remeshes."""
    from collections import OrderedDict
    from . import mesh_cache
    ids = tuple(sorted(session.selected - session.hidden if ids is None else ids))
    if hasattr(session.dataset,"scene_geometry"):
        return session.dataset.scene_geometry(session,ids,stride)
    key = (session.revision, ids, stride)
    if not hasattr(session, '_scene_mesh_cache'):
        session._scene_mesh_cache = OrderedDict()
    cache = session._scene_mesh_cache
    if key in cache:
        session._scene_geometry_sources=['memory']
        cache.move_to_end(key)
        return cache[key]
    from skimage.measure import marching_cubes
    grid = session.dataset.atoms
    result = []
    sources=[]
    for sid in ids:
        _, atoms = session._segment(sid)
        content_key=mesh_cache.key_for(session.dataset,atoms,stride)
        single_key=('geometry',sid,content_key)
        if single_key in cache:
            cache.move_to_end(single_key)
            result.extend(cache[single_key])
            sources.append('memory')
            continue
        disk_path=mesh_cache.path_for(session.dataset,atoms,stride,key=content_key)
        saved=mesh_cache.load(disk_path)
        if saved is not None:
            part=(sid,*saved);cache[single_key]=[part];result.append(part)
            sources.append('disk')
            continue
        sources.append('extracted')
        if hasattr(session.dataset, 'mesh_parts'):
            parts = session.dataset.mesh_parts(atoms, stride)
        else:
            sampled=grid.data[::stride,::stride,::stride]
            mask = np.isin(sampled, np.asarray(sorted(atoms), dtype=grid.data.dtype))
            if not mask.any():
                raise ValueError(f'Segment {sid} vanished at stride {stride}; reduce stride')
            v,f,_,_ = marching_cubes(np.pad(mask, 1).astype(np.float32), .5)
            v = grid.origin + (v-1)*grid.resolution*stride + grid.resolution/2
            parts = [(v,f)]
        vertices, faces, count = [], [], 0
        for v,f in parts:
            vertices.append(v); faces.append(f+count); count += len(v)
        if not vertices:
            raise ValueError(f'Segment {sid} vanished at stride {stride}; reduce stride')
        part=(sid,np.concatenate(vertices),np.concatenate(faces))
        mesh_cache.save(disk_path,part[1],part[2])
        cache[single_key]=[part]
        result.append(part)
    session._scene_geometry_sources=sources
    cache[key] = result
    def size(items):
        arrays={id(a):a for meshes in items for _,v,f in meshes for a in (v,f)}
        return sum(a.nbytes for a in arrays.values())
    # Always retain the current selection; older selections fit within 256 MiB.
    while len(cache)>1 and size(cache.values())>256*1024**2:
        cache.popitem(last=False)
    return result


def validate(camera):
    vector(camera['target_nm'], 'target_nm')
    if not np.isfinite([camera['azimuth'],camera['elevation'],camera['view_height_nm']]).all() or camera['view_height_nm']<=0:
        raise ValueError('Finite camera angles and positive view_height_nm required')
    if not -89.9 <= camera['elevation'] <= 89.9:
        raise ValueError('elevation must be between -89.9 and 89.9 degrees')
    return camera


def render_scene(session, output):
    """Full-resolution, depth-tested rendering with persistent graphics buffers."""
    import time
    from .raster import render
    started=time.perf_counter()
    meshes=geometry(session)
    extracted=time.perf_counter()
    v=session.view
    plane=[v.xyz(x,y) for x,y in [(-.5,-.5),(v.width-.5,-.5),(v.width-.5,v.height-.5),(-.5,v.height-.5),(-.5,-.5)]]
    bounds=session.dataset.scene_bounds(session) if hasattr(session.dataset,"scene_bounds") else (session.dataset.atoms.origin,session.dataset.atoms.upper)
    timings=render(meshes,validate(session.scene),bounds,plane,output)
    timings.update(geometry_seconds=extracted-started,total_seconds=time.perf_counter()-started,
                   triangles=sum(len(f) for _,_,f in meshes),stride=1,
                   geometry_sources=session._scene_geometry_sources)
    return timings


def write_glb(session, output):
    """GLB coordinates are micrometres relative to the volume origin."""
    blob=bytearray();views=[];accessors=[];meshes=[];nodes=[];materials=[]
    def accessor(a,kind,target):
        a=np.ascontiguousarray(a);offset=len(blob);blob.extend(a.tobytes())
        while len(blob)%4:blob.append(0)
        views.append({'buffer':0,'byteOffset':offset,'byteLength':a.nbytes,'target':target})
        item={'bufferView':len(views)-1,'componentType':5126 if a.dtype.kind=='f' else 5125,'count':len(a),'type':kind}
        if kind=='VEC3':item.update(min=a.min(axis=0).tolist(),max=a.max(axis=0).tolist())
        accessors.append(item);return len(accessors)-1
    for sid,v,f in geometry(session):
        v=((v-session.dataset.atoms.origin)/1000).astype('<f4');f=f.astype('<u4').ravel()
        vi=accessor(v,'VEC3',34962);fi=accessor(f,'SCALAR',34963)
        materials.append({'pbrMetallicRoughness':{'baseColorFactor':[*list(color(sid)/255),1.],'metallicFactor':0.,'roughnessFactor':.65},'doubleSided':True})
        meshes.append({'primitives':[{'attributes':{'POSITION':vi},'indices':fi,'material':len(materials)-1}]})
        nodes.append({'mesh':len(meshes)-1,'name':str(sid),'extras':{'segment_id':str(sid)}})
    doc={'asset':{'version':'2.0','generator':'Connectome selected segments'},'scene':0,'scenes':[{'nodes':list(range(len(nodes)))}],
         'nodes':nodes,'meshes':meshes,'materials':materials,'buffers':[{'byteLength':len(blob)}], 'bufferViews':views,'accessors':accessors,
         'extras':{'origin_nm':session.dataset.atoms.origin.tolist(),'units':'micrometres','revision':session.revision,'stride':1}}
    data=json.dumps(doc,separators=(',',':')).encode()
    data+=b' '*((-len(data))%4)
    total=12+8+len(data)+8+len(blob)
    with open(output,'wb') as f:
        f.write(struct.pack('<4sII',b'glTF',2,total));f.write(struct.pack('<I4s',len(data),b'JSON'));f.write(data)
        f.write(struct.pack('<I4s',len(blob),b'BIN\0'));f.write(blob)
