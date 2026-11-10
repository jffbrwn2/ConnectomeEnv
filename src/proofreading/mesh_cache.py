"""Optional bounded disk cache for exact meshes, shared across sessions/workers."""
import hashlib
import os
from pathlib import Path
import tempfile
import numpy as np

VERSION = 'marching-cubes-exact-v1'


def key_for(dataset, atoms, stride):
    h=hashlib.sha256()
    h.update((VERSION+'|'+getattr(dataset, 'geometry_fingerprint', dataset.fingerprint)+'|'+str(stride)).encode())
    h.update(np.asarray(sorted(atoms),dtype='<u8').tobytes())
    return h.hexdigest()


def path_for(dataset, atoms, stride, *, key=None):
    root=os.environ.get('CONNECTOME_MESH_CACHE')
    if not root:
        return None
    directory=Path(root);directory.mkdir(parents=True,exist_ok=True)
    return directory/((key or key_for(dataset,atoms,stride))+'.npz')


def load(path):
    if path is None:
        return None
    try:
        f=path.open('rb')
    except FileNotFoundError:
        return None
    with f, np.load(f,allow_pickle=False) as data:
        vertices=data['vertices'];faces=data['faces']
    if vertices.ndim!=2 or vertices.shape[1]!=3 or faces.ndim!=2 or faces.shape[1]!=3:
        raise ValueError('Invalid cached mesh: '+str(path))
    # Another writer may evict the path while this reader holds its open file.
    try:
        os.utime(path,None)
    except FileNotFoundError:
        pass
    return vertices,faces


def save(path, vertices, faces):
    if path is None:
        return
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.mesh-',delete=False) as f:
        temporary=Path(f.name)
        try:
            np.savez(f,vertices=vertices,faces=faces)
            f.flush()
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)
    # Access-time LRU using modification times; readers hold open files safely.
    files=[]
    for item in path.parent.glob('*.npz'):
        try:
            stat=item.stat();files.append((stat.st_mtime,stat.st_size,item))
        except FileNotFoundError:
            continue
    total=sum(n for _,n,_ in files)
    for _,size,item in sorted(files):
        if total<=2*1024**3:
            break
        if item==path:
            continue
        item.unlink(missing_ok=True);total-=size
