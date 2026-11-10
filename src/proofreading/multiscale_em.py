"""Explicit, bounded source-resolution EM sampling in physical coordinates."""
from collections import OrderedDict
import threading
import numpy as np
from .volume import Grid


class MultiscaleEM:
    """Keep the original grid as default; finer levels require an explicit action.

    Source metadata is pinned in the dataset manifest. Cutouts have a bounded
    voxel count even for oblique views; no whole-volume materialization.
    """
    def __init__(self, original, config, loader=None):
        self.original = original
        self.origin, self.upper, self.resolution = original.origin, original.upper, original.resolution
        self.source, self.info = config['source'], config['info']
        self.levels = {float(s['resolution'][0]): i for i, s in enumerate(self.info['scales'])
                       if np.all(np.asarray(s['resolution']) <= self.resolution)}
        self.loader = loader
        self.clients, self.cache = {}, OrderedDict()
        self.cache_bytes = 0
        self.max_cache_bytes = 128 * 1024**2
        self.max_cutout_voxels = 1024**2
        self.lock = threading.RLock()
        self.streaming = bool(config.get('streaming', False))
        self.streaming_grids = {}
        if self.streaming:
            from .streaming import ChunkCache
            import os
            self.chunk_cache = ChunkCache(directory=os.environ.get('CONNECTOME_IMAGE_CACHE'))

    def indices(self, xyz):
        return self.original.indices(xyz)

    def for_view(self, view):
        requested = view.em_resolution_nm
        if requested is None or requested == float(self.resolution[0]):
            return self.original
        if requested not in self.levels:
            raise ValueError(f'EM resolution must be one of {sorted(self.levels)} nm XY')
        return SourceLevel(self, self.levels[requested])

    def describe(self, view):
        grid = self.for_view(view)
        return {'resolution_nm': grid.resolution.tolist(),
                'available_xy_nm': sorted(self.levels),
                'default_resolution_nm': self.resolution.tolist(),
                'source': self.source, 'sampling': 'trilinear',
                'backend': 'original-grid' if grid is self.original else 'source-pyramid',
                'zoom_changes_resolution': False}

    def streamed_grid(self, mip):
        from .streaming import StreamingGrid
        if mip not in self.streaming_grids:
            scale = self.info['scales'][mip]
            self.streaming_grids[mip] = StreamingGrid(
                {'source':self.source,'info':self.info,'mip':mip},
                scale['resolution'],scale['voxel_offset'],scale['size'],
                self.info['data_type'],lambda lo,hi:self.download(mip,lo,hi),
                cache=self.chunk_cache)
        return self.streaming_grids[mip]

    def download(self, mip, lo, hi):
        if self.loader is not None:
            return self.loader(mip,lo,hi)
        from cloudvolume import CloudVolume
        if mip not in self.clients:
            self.clients[mip] = CloudVolume(
                self.source,mip=mip,info=self.info,bounded=True,
                fill_missing=False,progress=False,parallel=False,
                cache=False,lru_bytes=16*1024**2)
        try:
            return self.clients[mip][tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))]
        except Exception as exc:
            raise ValueError(f'EM source download failed at mip {mip}: {type(exc).__name__}') from exc

    def cutout(self, mip, lo, hi):
        key = (mip, tuple(lo), tuple(hi))
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            data = self.download(mip,lo,hi)
            data = np.asarray(data)
            if data.ndim == 4 and data.shape[-1] == 1:
                data = data[..., 0]
            if tuple(data.shape) != tuple(hi-lo):
                raise ValueError('EM source returned an unexpected cutout shape')
            data = np.ascontiguousarray(data)
            while self.cache and self.cache_bytes + data.nbytes > self.max_cache_bytes:
                _, old = self.cache.popitem(last=False)
                self.cache_bytes -= old.nbytes
            if data.nbytes <= self.max_cache_bytes:
                self.cache[key] = data
                self.cache_bytes += data.nbytes
            return data


class SourceLevel:
    def __init__(self, parent, mip):
        self.parent, self.mip = parent, mip
        scale = parent.info['scales'][mip]
        self.resolution = np.asarray(scale['resolution'], float)
        self.offset = np.asarray(scale['voxel_offset'], dtype=np.int64)
        self.end = self.offset + np.asarray(scale['size'], dtype=np.int64)

    def indices(self, xyz):
        _, valid = self.parent.indices(xyz)
        xyz = np.asarray(xyz)
        indices = np.floor(xyz / self.resolution).astype(np.int64) - self.offset
        valid &= ((xyz >= self.offset*self.resolution) & (xyz < self.end*self.resolution)).all(-1)
        return indices, valid

    def linear(self, xyz):
        if self.parent.streaming:
            values,valid = self.parent.streamed_grid(self.mip).linear(xyz)
            _,inside = self.parent.indices(xyz)
            valid &= inside
            return np.where(valid,values,0),valid
        xyz = np.asarray(xyz)
        result = np.zeros(xyz.shape[:-1], float)
        _, valid = self.indices(xyz)
        # Work on small screen tiles. Recursion also bounds extreme zoom-out
        # and oblique bounding boxes at explicitly requested native resolution.
        flat = xyz.reshape(-1, 3)
        indices = np.flatnonzero(valid)
        out = result.reshape(-1)
        def sample(ids):
            if not len(ids):
                return
            points = flat[ids]
            voxel = points/self.resolution - .5
            lo = np.maximum(np.floor(voxel.min(0)).astype(np.int64), self.offset)
            hi = np.minimum(np.floor(voxel.max(0)).astype(np.int64)+2, self.end)
            if np.prod(hi-lo, dtype=np.float64) > self.parent.max_cutout_voxels:
                axis = np.argmax(np.ptp(voxel, axis=0))
                order = np.argsort(voxel[:,axis], kind='stable')
                mid = len(ids)//2
                sample(ids[order[:mid]])
                sample(ids[order[mid:]])
                return
            data = self.parent.cutout(self.mip, lo, hi)
            grid = Grid(data, self.resolution, lo*self.resolution)
            # Grid validity is intentionally ignored here: at source edges,
            # interpolation clamps to the first/last voxel center.
            from scipy.ndimage import map_coordinates
            out[ids] = map_coordinates(grid.data, ((points-grid.origin)/self.resolution-.5).T,
                                       order=1, mode='nearest', output=float, prefilter=False)
        for start in range(0, len(indices), 4096):
            sample(indices[start:start+4096])
        return result, valid


def sampling_metadata(em, view):
    if hasattr(em, 'describe'):
        return em.describe(view)
    if view.em_resolution_nm is not None and view.em_resolution_nm != float(em.resolution[0]):
        raise ValueError('This dataset only has its original EM resolution')
    return {'resolution_nm': em.resolution.tolist(), 'available_xy_nm': [float(em.resolution[0])],
            'backend': 'original-grid', 'zoom_changes_resolution': False}
