"""Bounded, on-demand XYZ sampling without a materialized volume.

The loader receives source voxel bounds, never a requested screen bounding box.
Chunk identity includes the pinned source descriptor supplied by the caller.
This module handles public imaging only; it does not infer segment ownership.
"""
from collections import OrderedDict
from concurrent.futures import Future
from pathlib import Path
import hashlib
import json
import os
import threading
import uuid
import numpy as np
from .volume import vector


class ChunkCache:
    """Byte-bounded RAM and optional disk cache, shared by source grids.

    Independent misses overlap within a bounded fetch budget; identical misses
    share one result. Cached reads do not wait for unrelated network requests. Exceptions are not cached; missing data never become zero labels.
    """
    def __init__(self, max_bytes=128*1024**2, directory=None, disk_bytes=2*1024**3):
        if max_bytes <= 0 or disk_bytes < 0:
            raise ValueError('Cache budgets must be positive RAM and nonnegative disk bytes')
        self.max_bytes, self.disk_bytes = int(max_bytes), int(disk_bytes)
        self.directory = Path(directory) if directory is not None else None
        self.memory = OrderedDict(); self.resident_bytes = 0
        self.hits = self.disk_hits = self.misses = self.download_bytes = 0
        self.lock = threading.RLock()
        self.inflight = {}
        self.fetch_slots = threading.BoundedSemaphore(4)
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._trim_disk()

    def _trim_disk(self):
        files=[];total=0
        # DirEntry avoids constructing and resolving a Path for every cache file.
        # Rescan on each write so independent writers share the same disk bound.
        with os.scandir(self.directory) as entries:
            for entry in entries:
                if not (entry.name.startswith('chunk-') and entry.name.endswith('.npy')):continue
                try:stat=entry.stat()
                except FileNotFoundError:continue
                files.append((stat.st_mtime_ns,entry.path,stat.st_size));total+=stat.st_size
        if total<=self.disk_bytes:return
        for _,path,n in sorted(files):
            if total<=self.disk_bytes:break
            try:os.unlink(path)
            except FileNotFoundError:pass
            total-=n

    @staticmethod
    def digest(key, shape, dtype):
        return hashlib.sha256(json.dumps([key,tuple(map(int,shape)),np.dtype(dtype).str],sort_keys=True).encode()).hexdigest()

    def contains(self, key, shape, dtype):
        digest=self.digest(key,shape,dtype)
        with self.lock:
            return digest in self.memory or (self.directory is not None and
                    (self.directory/f'chunk-{digest}.npy').exists())

    def get(self, key, shape, dtype, loader):
        dtype = np.dtype(dtype); shape = tuple(map(int,shape))
        expected = int(np.prod(shape))*dtype.itemsize
        if expected > self.max_bytes:
            raise ValueError('One source chunk exceeds RAM budget; configure smaller chunks')
        digest = self.digest(key,shape,dtype)
        with self.lock:
            if digest in self.memory:
                self.hits += 1; self.memory.move_to_end(digest)
                return self.memory[digest]
            pending=self.inflight.get(digest)
            owner=pending is None
            if owner:
                pending=Future();self.inflight[digest]=pending
        if not owner:return pending.result()
        try:
            with self.fetch_slots:
                path=self.directory/f'chunk-{digest}.npy' if self.directory is not None else None
                array=None;disk_hit=False
                if path is not None:
                    try:
                        array=np.load(path,allow_pickle=False)
                        if array.shape!=shape or array.dtype!=dtype:raise ValueError('Corrupt cached chunk')
                    except (OSError,ValueError):
                        path.unlink(missing_ok=True);array=None
                    else:
                        disk_hit=True
                        try:os.utime(path,None)
                        except FileNotFoundError:pass
                downloaded=array is None
                if downloaded:
                    array=np.asarray(loader())
                    if array.ndim==4 and array.shape[-1]==1:array=array[...,0]
                    if array.shape!=shape or array.dtype!=dtype:
                        raise ValueError('Source chunk shape/dtype differs from pinned metadata')
                    if array.dtype.kind=='f' and not np.isfinite(array).all():
                        raise ValueError('Nonfinite source intensities')
                    # Own the allocation: a small slice must not retain a large
                    # batched download behind the byte-budget accounting.
                    array=np.array(array,copy=True,order='C')
                    if path is not None and array.nbytes+256<=self.disk_bytes:
                        temporary=path.with_suffix('.'+uuid.uuid4().hex+'.tmp')
                        try:
                            with temporary.open('wb') as f:np.save(f,array,allow_pickle=False)
                            temporary.replace(path)
                            with self.lock:self._trim_disk()
                        finally:temporary.unlink(missing_ok=True)
                array.flags.writeable=False
                with self.lock:
                    self.disk_hits+=int(disk_hit)
                    self.misses+=int(downloaded)
                    self.download_bytes+=array.nbytes if downloaded else 0
                    while self.memory and self.resident_bytes+array.nbytes>self.max_bytes:
                        _,old=self.memory.popitem(last=False);self.resident_bytes-=old.nbytes
                    self.memory[digest]=array;self.resident_bytes+=array.nbytes
                    self.inflight.pop(digest)
                    pending.set_result(array)
                return array
        except BaseException as exc:
            with self.lock:
                self.inflight.pop(digest,None);pending.set_exception(exc)
            raise

    def stats(self):
        with self.lock:
            return {'resident_bytes':self.resident_bytes,'ram_budget_bytes':self.max_bytes,
                    'memory_hits':self.hits,'disk_hits':self.disk_hits,
                    'source_requests':self.misses,'download_bytes':self.download_bytes}


class StreamingGrid:
    """Grid protocol for point/slice sampling over the entire source extent.

    No ``data`` array is exposed: whole-volume algorithms must explicitly adopt
    streaming rather than accidentally materializing the source. Global source
    indices include the source voxel offset; output coordinates remain XYZ nm.
    """
    def __init__(self, source_identity, resolution, offset, shape, dtype, loader,
                 cache=None, chunk_shape=(64,64,16), on_labels=None, fetch_pool=None):
        self.identity = hashlib.sha256(json.dumps([source_identity,list(map(int,chunk_shape))],sort_keys=True).encode()).hexdigest()
        self.resolution = vector(resolution,'resolution',True)
        self.offset = np.asarray(offset,dtype=np.int64)
        self.shape = np.asarray(shape,dtype=np.int64)
        self.chunk_shape = np.asarray(chunk_shape,dtype=np.int64)
        if any(a.shape!=(3,) for a in (self.offset,self.shape,self.chunk_shape)) or np.any(self.shape<=0) or np.any(self.chunk_shape<=0):
            raise ValueError('Shape, offset and chunk shape must be XYZ; sizes must be positive')
        self.dtype = np.dtype(dtype)
        if self.dtype.kind not in 'uif':raise ValueError('Numeric imaging data required')
        self.origin = self.offset*self.resolution
        self.upper = (self.offset+self.shape)*self.resolution
        self.loader, self.cache = loader, cache if cache is not None else ChunkCache()
        self.on_labels = on_labels
        self.fetch_pool = fetch_pool
        self.lock = threading.RLock()
        self.source_calls = 0
        self.source_bytes = 0

    def indices(self, xyz):
        xyz = np.asarray(xyz,dtype=float)
        if xyz.shape[-1:]!=(3,) or not np.isfinite(xyz).all():raise ValueError('Finite XYZ coordinates required')
        index = np.floor((xyz-self.origin)/self.resolution).astype(np.int64)
        return index, ((index>=0)&(index<self.shape)).all(axis=-1)

    def _download(self, lo, hi):
        data=np.asarray(self.loader(lo,hi))
        self.source_calls+=1;self.source_bytes+=data.nbytes
        return data

    def _prefetch(self, indices, halo=0):
        with self.lock:
            return self._prefetch_locked(indices,halo)

    def _prefetch_locked(self, indices, halo=0):
        if not len(indices):return True
        dimensions=(self.shape+self.chunk_shape-1)//self.chunk_shape
        keys=np.unique(np.ravel_multi_index((indices//self.chunk_shape).T,dimensions))
        chunks=np.asarray(np.unravel_index(keys,dimensions)).T
        pending=[]
        for chunk in chunks:
            lo=np.maximum(chunk*self.chunk_shape-halo,0)
            hi=np.minimum((chunk+1)*self.chunk_shape+halo,self.shape)
            key=[self.identity,chunk.tolist()]+(['linear-halo'] if halo else [])
            if not self.cache.contains(key,hi-lo,self.dtype):pending.append((key,lo,hi))
        if sum(int(np.prod(b-a))*self.dtype.itemsize for _,a,b in pending)>self.cache.max_bytes//2:
            return False  # Per-screen-block gathers avoid prefetch eviction/thrashing.
        batches=[]
        def plan(items):
            if not items:return
            lo=np.min([r[1] for r in items],axis=0);hi=np.max([r[2] for r in items],axis=0)
            budget=min(32*1024**2,self.cache.max_bytes)
            if int(np.prod(hi-lo))*self.dtype.itemsize>budget and len(items)>1:
                centers=np.asarray([r[1]+r[2] for r in items]);axis=np.argmax(np.ptp(centers,axis=0))
                order=np.argsort(centers[:,axis],kind='stable');mid=len(items)//2
                plan([items[i] for i in order[:mid]]);plan([items[i] for i in order[mid:]])
                return
            if int(np.prod(hi-lo))*self.dtype.itemsize>self.cache.max_bytes:
                raise ValueError('One source chunk exceeds RAM budget')
            batches.append((items,lo,hi))
        def fetch(batch):
            items,lo,hi=batch
            # Coalesce neighboring cache misses into one bounded source request.
            # CloudVolume can then batch its underlying object-store reads.
            data=self._download(lo+self.offset,hi+self.offset)
            if data.ndim==4 and data.shape[-1]==1:data=data[...,0]
            if data.shape!=tuple(hi-lo) or data.dtype!=self.dtype:
                raise ValueError('Source chunk shape/dtype differs from pinned metadata')
            for key,a,b in items:
                slices=tuple(slice(int(x),int(y)) for x,y in zip(a-lo,b-lo))
                self.cache.get(key,b-a,self.dtype,lambda slices=slices:data[slices])
        plan(pending)
        if self.fetch_pool is not None and len(batches)>1:
            futures=[self.fetch_pool.submit(fetch,b) for b in batches]
            from concurrent.futures import wait
            wait(futures)  # Settle every fetch before a failed action rolls back.
            for future in futures:future.result()
        else:
            for batch in batches:fetch(batch)
        return True

    def read_box(self, lo, hi):
        """Read a bounded global-voxel box through the same chunk cache as views."""
        lo=np.asarray(lo,dtype=np.int64)-self.offset
        hi=np.asarray(hi,dtype=np.int64)-self.offset
        if lo.shape!=(3,) or hi.shape!=(3,) or np.any(lo<0) or np.any(hi>self.shape) or np.any(hi<=lo):
            raise ValueError('Box must be nonempty and inside the source grid')
        if int(np.prod(hi-lo))*self.dtype.itemsize>32*1024**2:
            raise ValueError('One volume tile exceeds the 32 MiB read budget')
        axes=[np.arange(a,b+1) for a,b in zip(lo//self.chunk_shape,(hi-1)//self.chunk_shape)]
        chunks=np.stack(np.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3)
        self._prefetch(chunks*self.chunk_shape)
        output=np.empty(tuple(hi-lo),self.dtype)
        for chunk in chunks:
            start=chunk*self.chunk_shape;stop=np.minimum(start+self.chunk_shape,self.shape)
            key=[self.identity,chunk.tolist()]
            data=self.cache.get(key,stop-start,self.dtype,
                lambda a=start,b=stop:self._download(a+self.offset,b+self.offset))
            a=np.maximum(start,lo);b=np.minimum(stop,hi)
            output[tuple(slice(int(x),int(y)) for x,y in zip(a-lo,b-lo))]=data[
                tuple(slice(int(x),int(y)) for x,y in zip(a-start,b-start))]
        return output

    def _values(self, indices, *, prefetch=True):
        """Gather only intersected chunks; no bounding-box download for oblique views."""
        if prefetch:self._prefetch(indices)
        result = np.empty(len(indices),dtype=self.dtype)
        chunk_ids = indices//self.chunk_shape
        dimensions = (self.shape+self.chunk_shape-1)//self.chunk_shape
        keys = np.ravel_multi_index(chunk_ids.T,dimensions)
        unique,inverse = np.unique(keys,return_inverse=True)
        # Group once instead of scanning the entire image for every chunk.
        order = np.argsort(inverse,kind='stable')
        boundaries = np.r_[0,np.cumsum(np.bincount(inverse,minlength=len(unique)))]
        for i,key in enumerate(unique):
            chunk = np.asarray(np.unravel_index(key,dimensions))
            positions = order[boundaries[i]:boundaries[i+1]]
            lo = chunk*self.chunk_shape; hi = np.minimum(lo+self.chunk_shape,self.shape)
            a,b = lo+self.offset,hi+self.offset
            data = self.cache.get([self.identity,chunk.tolist()],hi-lo,self.dtype,
                                  lambda a=a,b=b:self._download(a,b))
            relative = indices[positions]-lo
            result[positions] = data[tuple(relative.T)]
        return result

    def _sample_blocks(self, indices, selected, *, halo=0, ready=False):
        """Consume cache-sized groups of whole chunks without per-point refetches."""
        if not len(selected):return
        if ready:
            yield selected
            return
        dimensions=(self.shape+self.chunk_shape-1)//self.chunk_shape
        keys=np.ravel_multi_index((indices[selected]//self.chunk_shape).T,dimensions)
        order=np.argsort(keys,kind='stable');selected=selected[order];keys=keys[order]
        boundaries=np.r_[0,np.flatnonzero(np.diff(keys))+1,len(keys)]
        first=0;size=0
        for i in range(len(boundaries)-1):
            chunk=np.asarray(np.unravel_index(keys[boundaries[i]],dimensions))
            lo=np.maximum(chunk*self.chunk_shape-halo,0)
            hi=np.minimum((chunk+1)*self.chunk_shape+halo,self.shape)
            chunk_bytes=int(np.prod(hi-lo))*self.dtype.itemsize
            if size and size+chunk_bytes>self.cache.max_bytes//2:
                block=selected[boundaries[first]:boundaries[i]]
                self._prefetch(indices[block],halo=halo)
                yield block
                first=i;size=0
            size+=chunk_bytes
        block=selected[boundaries[first]:]
        self._prefetch(indices[block],halo=halo)
        yield block

    def nearest(self, xyz):
        indices,valid = self.indices(xyz)
        flat = indices.reshape(-1,3); take = np.flatnonzero(valid.reshape(-1))
        out = np.zeros(len(flat),dtype=self.dtype)
        ready=self._prefetch(flat[take])
        for start in range(0,len(take),65536):
            for selected in self._sample_blocks(flat,take[start:start+65536],ready=ready):
                out[selected] = self._values(flat[selected],prefetch=False)
        if self.on_labels is not None:self.on_labels(np.unique(out[out!=0]))
        return out.reshape(valid.shape),valid

    def linear(self, xyz):
        from scipy.ndimage import map_coordinates
        xyz = np.asarray(xyz,float); _,valid = self.indices(xyz)
        points = ((xyz-self.origin)/self.resolution-.5).reshape(-1,3)
        take = np.flatnonzero(valid.reshape(-1)); out = np.zeros(len(points),float)
        anchors=np.clip(np.floor(points).astype(np.int64),0,self.shape-1)
        ready=self._prefetch(anchors[take],halo=1)
        # Interpolate each source chunk in compiled scipy code. A one-voxel
        # halo replaces eight separately sorted Python gathers per sample.
        for start in range(0,len(take),65536):
            for selected in self._sample_blocks(anchors,take[start:start+65536],halo=1,ready=ready):
                p = points[selected]
                anchor = anchors[selected]
                chunks = anchor//self.chunk_shape
                dimensions = (self.shape+self.chunk_shape-1)//self.chunk_shape
                keys = np.ravel_multi_index(chunks.T,dimensions)
                unique,inverse = np.unique(keys,return_inverse=True)
                order = np.argsort(inverse,kind='stable')
                boundaries = np.r_[0,np.cumsum(np.bincount(inverse,minlength=len(unique)))]
                for i,key in enumerate(unique):
                    positions = order[boundaries[i]:boundaries[i+1]]
                    chunk = np.asarray(np.unravel_index(key,dimensions))
                    lo = np.maximum(chunk*self.chunk_shape-1,0)
                    hi = np.minimum((chunk+1)*self.chunk_shape+1,self.shape)
                    a,b = lo+self.offset,hi+self.offset
                    data = self.cache.get([self.identity,chunk.tolist(),'linear-halo'],hi-lo,self.dtype,
                                          lambda a=a,b=b:self._download(a,b))
                    out[selected[positions]] = map_coordinates(data,(p[positions]-lo).T,
                        order=1,mode='nearest',output=float,prefilter=False)
        return out.reshape(valid.shape),valid
