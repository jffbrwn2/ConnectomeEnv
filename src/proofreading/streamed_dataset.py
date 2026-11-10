"""Explorable source-backed environment with sparse, persistent edit rules.

Source extents govern navigation. Evaluation scope is deliberately not loaded by
this backend. Production source ownership must be pinned to an explicit time.
"""
from collections import OrderedDict
from datetime import datetime
import hashlib,json,os,time,uuid
from pathlib import Path
import numpy as np
from .streamed_sources import StreamedSources
from .streaming import ChunkCache
from .streaming_partition import StreamingPartition
from .ownership_cache import OwnershipCache


def trace(event, **fields):
    if os.environ.get('CONNECTOME_TRACE_STREAMING')=='1':
        print(json.dumps({'streaming':event,'monotonic':time.monotonic(),**fields}),flush=True)


class CaveSource:
    def __init__(self, config, cache_directory):
        from caveclient import CAVEclient
        from cloudvolume import CloudVolume
        self.config=config;self.directory=Path(cache_directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.mesh_directory=Path(config['cache_directory'])/'source-meshes'
        self.mesh_directory.mkdir(parents=True,exist_ok=True)
        self._valid_roots=OrderedDict()
        self.timestamp=datetime.fromisoformat(config['timestamp'].replace('Z','+00:00'))
        if self.timestamp.tzinfo is None:raise ValueError('Source timestamp must include timezone')
        trace('source_initializing')
        self.client=CAVEclient(config['datastack'], auth_token=os.environ.get('CAVE_API_TOKEN'))
        self.volumes={'em':CloudVolume(config['em']['source'],info=config['em']['info'],progress=False,
                              parallel=False,cache=False,lru_bytes=32*1024**2),
                      'segmentation':self.client.info.segmentation_cloudvolume(agglomerate=False,
                              progress=False,parallel=False,cache=False,lru_bytes=32*1024**2)}
        seg=self.volumes['segmentation']
        if seg.info['scales']!=config['segmentation']['info']['scales']:
            raise ValueError('Segmentation source metadata changed from pinned manifest')
        self.ownership=OwnershipCache({'datastack':config['datastack'],'timestamp':config['timestamp'],
                    'source':config['segmentation']['source']},
                    self.resolve,self.directory/'ownership',resolver_workers=4)
        self.ownership.db.execute('CREATE INDEX IF NOT EXISTS owners_root ON owners(root)')

    def resolve(self, atoms):
        trace('ownership_start',atoms=len(atoms))
        try:roots=self.client.chunkedgraph.get_roots(atoms,timestamp=self.timestamp)
        except Exception as exc:raise ValueError(f'Pinned ownership fetch failed: {type(exc).__name__}') from exc
        trace('ownership_end',atoms=len(atoms))
        return roots

    def download(self, channel, mip, lo, hi):
        from cloudvolume import Bbox
        try:
            trace('download_start',channel=channel,mip=mip,shape=list(map(int,hi-lo)))
            # Explicit coordinates avoid CloudVolume validating against its
            # default mip's bounds; never mutate shared volume.mip across threads.
            data=np.asarray(self.volumes[channel].download(Bbox(lo,hi),mip=mip,agglomerate=False,
                              coord_resolution=self.volumes[channel].meta.resolution(mip)))
            trace('download_end',channel=channel,mip=mip)
            return data
        except Exception as exc:
            raise ValueError(f'{channel} source fetch failed: {type(exc).__name__}') from exc

    def ensure(self, atoms):return self.ownership.ensure(atoms)
    def observe(self, atoms):return self.ownership.observe(atoms)
    def owner(self, atom):return self.ownership.owner(atom)

    def observed(self, root):
        with self.ownership.lock:
            return [int(a) for (a,) in self.ownership.db.execute('SELECT atom FROM owners WHERE root=?',(str(root),))]

    def validate_root(self, root):
        root=int(root)
        if not 0 < root < 2**64:
            raise ValueError('Root ID must be a positive uint64')
        if root in self._valid_roots:
            self._valid_roots.move_to_end(root);return
        valid=np.asarray(self.client.chunkedgraph.is_latest_roots([root],timestamp=self.timestamp))
        if valid.shape!=(1,) or valid.dtype.kind!='b' or not valid[0]:
            raise ValueError('Requested root is not live at the pinned timestamp')
        self._valid_roots[root]=True
        while len(self._valid_roots)>1024:self._valid_roots.popitem(last=False)

    def members(self, root):
        # Root IDs are immutable historical nodes, not a request for latest roots.
        path=self.directory/f'members-{int(root)}.npy'
        if path.exists():return np.load(path,mmap_mode='r',allow_pickle=False)
        try:atoms=np.unique(np.asarray(self.client.chunkedgraph.get_leaves(int(root),stop_layer=1),np.uint64))
        except Exception as exc:raise ValueError(f'Root membership fetch failed: {type(exc).__name__}') from exc
        result=self.ensure(atoms)
        if any(r!=int(root) for r in result.values()):
            raise ValueError('Requested root is not live at the pinned timestamp')
        temporary=path.with_suffix('.'+uuid.uuid4().hex+'.partial')
        try:
            with temporary.open('wb') as f:np.save(f,atoms,allow_pickle=False)
            temporary.replace(path)
        finally:temporary.unlink(missing_ok=True)
        return atoms

    def mesh(self, root, bounds):
        from cloudvolume import Bbox,Mesh
        from . import mesh_cache
        key=hashlib.sha256(json.dumps(['source-root-render-fragments-v2',self.config['segmentation'],int(root),np.asarray(bounds).tolist()],sort_keys=True).encode()).hexdigest()
        path=self.mesh_directory/(key+'.npz')
        cached=mesh_cache.load(path)
        if cached is not None:return cached
        resolution=np.asarray(self.config['segmentation']['info']['scales'][0]['resolution'])
        box=Bbox(np.floor(bounds[0]/resolution).astype(int),np.ceil(bounds[1]/resolution).astype(int))
        try:
            trace('mesh_start',root=str(root))
            source=self.volumes['segmentation'].mesh
            if callable(getattr(source,'get_meshes_via_manifest',None)):
                # Rendering needs identical triangles, not welded vertex indices.
                # Keep the bounded historical manifest lookup; omit the expensive
                # cross-fragment deduplication performed even by get(..., False).
                fragments=source.get_meshes_via_manifest(int(root),box,use_byte_offsets=True)
                if not fragments:raise IndexError('Empty source mesh manifest')
                mesh=Mesh.concatenate(*fragments)
            else:
                mesh=source.get(int(root),bounding_box=box,
                               deduplicate_chunk_boundaries=False)[int(root)]
        except IndexError as exc:
            # Selecting a neuron outside the current 3D camera is valid. Check
            # emptiness explicitly; never disguise a missing in-view mesh.
            leaves=self.client.chunkedgraph.get_leaves(int(root),bounds=np.stack([box.minpt,box.maxpt],axis=1),stop_layer=1)
            if not len(leaves):return np.empty((0,3),np.float32),np.empty((0,3),np.uint32)
            raise ValueError('Source mesh missing for material inside the display window') from exc
        except Exception as exc:
            raise ValueError(f'Selected source mesh failed: {type(exc).__name__}') from exc
        vertices,faces=np.asarray(mesh.vertices),np.asarray(mesh.faces)
        if vertices.ndim!=2 or vertices.shape[1]!=3 or faces.ndim!=2 or faces.shape[1]!=3 or not np.isfinite(vertices).all():
            raise ValueError('Malformed source mesh geometry')
        if len(faces) and (faces.min()<0 or faces.max()>=len(vertices)):raise ValueError('Source mesh has invalid face indices')
        trace('mesh_end',root=str(root),vertices=len(vertices),faces=len(faces))
        mesh_cache.save(path,vertices,faces)
        return vertices,faces

    def mesh_fragments(self, root, bounds):
        """Keep layer-2 identity so edits can reuse every unaffected surface."""
        from cloudvolume import Bbox
        if not hasattr(self,'_fragment_cache'):self._fragment_cache=OrderedDict()
        key=(int(root),tuple(np.asarray(bounds).ravel()))
        if key in self._fragment_cache:
            self._fragment_cache.move_to_end(key);return self._fragment_cache[key]
        resolution=np.asarray(self.config['segmentation']['info']['scales'][0]['resolution'])
        box=Bbox(np.floor(bounds[0]/resolution).astype(int),np.ceil(bounds[1]/resolution).astype(int))
        try:
            fragments=self.volumes['segmentation'].mesh.get_meshes_via_manifest(int(root),box,use_byte_offsets=True)
        except Exception as exc:
            raise ValueError(f'Edited source fragment fetch failed: {type(exc).__name__}') from exc
        result=[(int(m.segid),np.asarray(m.vertices),np.asarray(m.faces)) for m in fragments]
        if sum(v.nbytes+f.nbytes for _,v,f in result)>128*1024**2:
            raise ValueError('Source fragments exceed display budget; zoom in')
        self._fragment_cache[key]=result
        while sum(v.nbytes+f.nbytes for parts in self._fragment_cache.values() for _,v,f in parts)>128*1024**2:
            self._fragment_cache.popitem(last=False)
        return result

    def fragment_members(self, root):
        # Historical layer-2 ownership is separate from the current root map.
        if not hasattr(self,'_fragment_members'):self._fragment_members={}
        root=int(root)
        if root not in self._fragment_members:
            atoms=np.asarray(self.members(root),np.uint64);groups={}
            for start in range(0,len(atoms),10000):
                part=atoms[start:start+10000]
                labels=np.asarray(self.client.chunkedgraph.get_roots(part,stop_layer=2,timestamp=self.timestamp))
                if labels.shape!=part.shape:raise ValueError('Malformed source fragment membership')
                for a,label in zip(part,labels):groups.setdefault(int(label),set()).add(int(a))
            self._fragment_members={root:groups}
        return self._fragment_members[root]

    def graph(self, roots):
        import networkx as nx
        graph=nx.Graph()
        from .source_graph import complete
        for root in roots:
            members=set(map(int,self.members(root)))
            graph.add_nodes_from(members)
            for edges,affinities in complete(self,root):
                for (a,b),weight in zip(edges,affinities):
                    a,b=int(a),int(b)
                    if a==b:continue
                    previous=graph.get_edge_data(a,b)
                    if previous is not None and previous['capacity']!=float(weight):
                        raise ValueError('Source returned conflicting duplicate edge capacities')
                    graph.add_edge(a,b,capacity=float(weight))
            if len(members)>1 and not nx.is_connected(graph.subgraph(members)):
                raise ValueError(f'Incomplete source graph for root {root}: disconnected historical root; split withheld')
        return graph


class StreamedGraph:
    def __init__(self,dataset):self.dataset=dataset
    def subgraph(self, atoms):
        d=self.dataset;ids=np.asarray(sorted(atoms),np.uint64)
        roots=set(d.source.ensure(ids).values())
        key=tuple(sorted(roots))
        if key not in d._graphs:
            # Keep only the most recent edit graph, rather than all explored roots.
            d._graphs.clear();d._graphs[key]=d.source.graph(roots)
        # Consumers copy only when editing topology (the cut's terminal nodes).
        # Component diagnostics need a view, not repeated million-node copies.
        d._graphs[key].add_nodes_from(atoms)
        return d._graphs[key].subgraph(atoms)


class StreamedDataset:
    partition_format='streaming-rules-v1'
    identity_kind='pinned-source-manifest'
    split_capacity='source agglomeration affinity capacity; infinite edges are protected'
    max_segment_id=2**63
    edges=True

    def __init__(self, manifest, source=None):
        self.manifest=Path(manifest).resolve();c=json.loads(self.manifest.read_text())
        if c.get('backend')!='streamed-world-v1':raise ValueError('Unsupported streamed backend')
        identity={k:v for k,v in c.items() if k!='cache_directory'}
        self.fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        self.geometry_fingerprint=self.fingerprint;self.name=c.get('name','streamed-world')
        cache_base=Path(os.environ.get('CONNECTOME_CACHE_DIRECTORY', c['cache_directory'])).expanduser()
        if not cache_base.is_absolute():cache_base=self.manifest.parent/cache_base
        c={**c,'cache_directory':str(cache_base)}
        source_key=hashlib.sha256(json.dumps({k:c.get(k) for k in ('datastack','timestamp','segmentation')},sort_keys=True).encode()).hexdigest()
        cache_dir=cache_base/source_key
        self.source=source if source is not None else CaveSource(c,cache_dir/'source')
        self.sources=StreamedSources(c,self.source.download,self.source,
                    ChunkCache(directory=cache_base/'chunks',max_bytes=128*1024**2))
        self.atoms,self.em=self.sources.atoms,self.sources.em
        self.initial=StreamingPartition(self.source)
        self.initial_view=c['initial_view']
        from concurrent.futures import ThreadPoolExecutor
        self._sampling_pool=ThreadPoolExecutor(max_workers=2,thread_name_prefix="source-view")
        self.observation_pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix="em-observation")
        self._graphs={};self._meshes=OrderedDict();self._native_meshes=OrderedDict()
        self._plane_cache=None

    def close(self):
        """Drain per-episode work before releasing cached data and source handles."""
        if getattr(self, '_closed', False):
            return
        self._closed = True
        self.observation_pool.shutdown(wait=True)
        self._sampling_pool.shutdown(wait=True)
        self.sources.fetch_pool.shutdown(wait=True)
        ownership = getattr(self.source, 'ownership', None)
        if ownership is not None:
            ownership.close()
        self._graphs.clear()
        self._meshes.clear()
        self._native_meshes.clear()
        self._plane_cache = None
        self.sources.cache.memory.clear()
        self.sources.cache.resident_bytes = 0

    def sample_plane(self, view, xyz):
        em,seg=self.sources.sampling_grids(view)
        # Cache source samples, not edited root labels or selection coloring.
        # Hash actual coordinates so arbitrary/oblique sampling cannot alias.
        xyz=np.ascontiguousarray(xyz)
        key=(em.identity,seg.identity,xyz.shape,xyz.dtype.str,hashlib.sha256(memoryview(xyz)).digest())
        if self._plane_cache is not None and self._plane_cache[0]==key:
            return self._plane_cache[1]
        intensity=self._sampling_pool.submit(em.linear,xyz)
        labels=self._sampling_pool.submit(seg.nearest,xyz)
        try:
            values,valid=intensity.result();atoms,inside=labels.result()
            result=(values,valid,atoms,inside)
            for array in result:array.flags.writeable=False
            self._plane_cache=(key,result) if sum(a.nbytes for a in result)<=64*1024**2 else None
            return result
        finally:
            # A failed fetch must settle its sibling before action rollback.
            from concurrent.futures import wait
            wait([intensity,labels])

    def sampling_grids(self, view):return self.sources.sampling_grids(view)
    def sampling_metadata(self, view):return self.sources.sampling_metadata(view)
    def graph(self):return StreamedGraph(self)

    def can_select_unobserved(self, partition, sid):
        # Original source roots can be displayed without enumerating their atoms.
        # Retired roots and locally edited children still use partition validation.
        if (not hasattr(self.source,'validate_root') or
            partition.roots_for(sid)!={int(sid)} or partition.atoms_for(sid)):
            return False
        self.source.validate_root(sid)
        return True

    def ensure_segment(self, partition, sid):
        if not partition.atoms_for(sid):
            for root in partition.roots_for(sid):
                members=self.source.members(root)
                if len(members):self.source.ensure(members[:1])

    def save_partition(self, partition, path):
        np.savez_compressed(path,format=np.asarray(self.partition_format),
             dataset_sha256=np.asarray(self.fingerprint),rules=np.asarray(json.dumps(partition.serialize(),sort_keys=True)))

    def scene_bounds(self, session):
        center=np.asarray(session.scene['target_nm']);height=session.scene['view_height_nm']
        # Rotation leaves this extent unchanged. It is a display window, not an
        # edit or navigation boundary. Round to stable physical blocks for reuse.
        extent=2**np.ceil(np.log2(max(height*2,256)))
        center=np.round(center/(extent/8))*(extent/8)
        lo=np.maximum(center-extent/2,self.sources.bounds[0])
        hi=np.minimum(center+extent/2,self.sources.bounds[1])
        return np.array([lo,hi])

    def scene_geometry(self, session, ids, stride):
        bounds=self.scene_bounds(session)
        if np.any(bounds[1]<=bounds[0]):
            session._scene_geometry_sources=['outside-source'];return []
        key=(id(session.partition.rules),tuple(ids),stride,tuple(bounds.ravel()))
        if key in self._meshes:
            self._meshes.move_to_end(key)
            session._scene_geometry_sources=['memory',*self._meshes[key][3]]
            return self._meshes[key][1]
        meshes=[];sources=[]
        for sid in ids:
            roots=session.partition.roots_for(sid)
            split_affected=bool(roots & session.partition.split_roots())
            if not split_affected:
                for root in sorted(roots):
                    native_key=(root,tuple(bounds.ravel()))
                    if native_key in self._native_meshes:
                        self._native_meshes.move_to_end(native_key)
                        vertices,faces=self._native_meshes[native_key]
                    else:
                        vertices,faces=self.source.mesh(root,bounds)
                        if vertices.nbytes+faces.nbytes>128*1024**2:
                            raise ValueError('Source mesh exceeds display budget; zoom in')
                        self._native_meshes[native_key]=(vertices,faces)
                        while sum(v.nbytes+f.nbytes for v,f in self._native_meshes.values())>128*1024**2:
                            self._native_meshes.popitem(last=False)
                    if len(vertices):meshes.append((sid,vertices,faces))
                sources.append('source-mesh')
            else:
                atomset=session.partition.complete_atoms_for(sid)
                if hasattr(self.source,'mesh_fragments') and hasattr(self.source,'fragment_members'):
                    for root in sorted(roots):
                        groups=self.source.fragment_members(root)
                        for fragment,vertices,faces in self.source.mesh_fragments(root,bounds):
                            if fragment not in groups:
                                raise ValueError('Source mesh fragment has no pinned membership')
                            members=groups[fragment];selected=members & atomset
                            if not selected:continue
                            if selected==members:
                                meshes.append((sid,vertices,faces))
                            else:
                                # Only a fragment crossed by the edit needs new geometry.
                                # Retain native voxels, including thin processes; never
                                # pick a coarser level to satisfy a display-size budget.
                                margin=np.maximum(self.atoms.resolution,128)
                                local=np.array([vertices.min(axis=0)-margin,vertices.max(axis=0)+margin])
                                v,f=self._edited_mesh(selected,local)
                                if len(v):meshes.append((sid,v,f))
                    sources.append('edited-fragments-native-'+str(self.sources.native[0])+'nm')
                else:
                    v,f=self._edited_mesh(atomset,bounds)
                    if len(v):meshes.append((sid,v,f))
                    sources.append('edited-label-native-'+str(self.sources.native[0])+'nm')
        size=sum(v.nbytes+f.nbytes for _,v,f in meshes)
        if size>256*1024**2:raise ValueError('Selected mesh exceeds cache budget; zoom in or reduce selection')
        self._meshes[key]=(session.partition.rules,meshes,size,tuple(sources))
        while sum(v[2] for v in self._meshes.values())>256*1024**2:self._meshes.popitem(last=False)
        session._scene_geometry_sources=sources
        return meshes

    def _edited_mesh(self, atomset, bounds):
        """Native-resolution tiled cells with a shared halo, cached by membership."""
        from skimage.measure import marching_cubes
        from . import mesh_cache
        grid=self.atoms
        lo=np.maximum(np.floor(bounds[0]/grid.resolution).astype(int),grid.offset)
        hi=np.minimum(np.ceil(bounds[1]/grid.resolution).astype(int),grid.offset+grid.shape)
        if np.any(hi<=lo):return np.empty((0,3),np.float32),np.empty((0,3),np.uint32)
        digest=hashlib.sha256(('edited-native-v1'+self.fingerprint+str((lo.tolist(),hi.tolist()))).encode())
        ids=np.asarray(sorted(atomset),dtype=np.uint64);digest.update(ids.tobytes())
        directory=Path(self.sources.config['cache_directory'])/'edited-meshes';directory.mkdir(parents=True,exist_ok=True)
        path=directory/(digest.hexdigest()+'.npz');cached=mesh_cache.load(path)
        if cached is not None:return cached
        vertices=[];faces=[];count=0;size=0
        # Tile marching-cubes cells, not independently padded voxel cubes.
        # Neighboring tiles share their boundary samples and produce no false caps.
        tile=np.array([128,128,32]);sample_lo=lo-1;sample_hi=hi+1
        for x in range(int(sample_lo[0]),int(sample_hi[0]-1),int(tile[0])):
            for y in range(int(sample_lo[1]),int(sample_hi[1]-1),int(tile[1])):
                for z in range(int(sample_lo[2]),int(sample_hi[2]-1),int(tile[2])):
                    start=np.array([x,y,z]);end=np.minimum(start+tile+1,sample_hi)
                    shape=tuple(end-start);mask=np.zeros(shape,dtype=np.float32)
                    a=np.maximum(start,lo);b=np.minimum(end,hi)
                    if np.any(b<=a):continue
                    labels=grid.read_box(a,b)
                    mask[tuple(slice(int(c),int(d)) for c,d in zip(a-start,b-start))]=np.isin(labels,ids)
                    if not mask.any() or mask.all():continue
                    v,f,_,_=marching_cubes(mask,.5)
                    v=((v+start)*grid.resolution+grid.resolution/2).astype(np.float32)
                    size+=v.nbytes+f.nbytes
                    if size>256*1024**2:raise ValueError('Edited mesh exceeds geometry budget; zoom in')
                    vertices.append(v);faces.append(f+count);count+=len(v)
        result=(np.concatenate(vertices) if vertices else np.empty((0,3),np.float32),
                np.concatenate(faces) if faces else np.empty((0,3),np.uint32))
        mesh_cache.save(path,*result)
        return result
