"""Paired whole-source imaging with lazy chunks and pinned ownership.

Exploration extent comes from the source pyramid, not an evaluation crop.
This is the imaging layer; the evaluator retains its separate frozen scope.
"""
import numpy as np
from .streaming import ChunkCache,StreamingGrid


class StreamedSources:
    def __init__(self, config, loader, ownership, cache=None):
        self.config=config;self.loader=loader;self.ownership=ownership
        self.cache=cache if cache is not None else ChunkCache()
        from concurrent.futures import ThreadPoolExecutor
        self.fetch_pool=ThreadPoolExecutor(max_workers=4,thread_name_prefix="source-chunks")
        self.levels={};self.grids={}
        for channel in ('em','segmentation'):
            info=config[channel]['info']
            self.levels[channel]={tuple(s['resolution']):i for i,s in enumerate(info['scales'])}
        self.shared=sorted(set(self.levels['em'])&set(self.levels['segmentation']))
        if not self.shared:raise ValueError('No physically matched source resolutions')
        # The existing action identifies levels by XY nanometers, so ambiguous
        # XY levels (different Z spacings at the same XY) must be rejected.
        if len({r[0] for r in self.shared})!=len(self.shared):
            raise ValueError('Ambiguous XY resolution in source pyramids')
        self.native=self.shared[0]
        self.em=self.level('em',self.native)
        self.atoms=self.level('segmentation',self.native)
        lo=np.maximum(self.em.origin,self.atoms.origin)
        hi=np.minimum(self.em.upper,self.atoms.upper)
        if np.any(hi<=lo):raise ValueError('EM and segmentation source extents do not overlap')
        self.bounds=(lo,hi)

    def level(self, channel, resolution):
        resolution=tuple(resolution);key=(channel,resolution)
        if key not in self.grids:
            descriptor=self.config[channel];info=descriptor['info']
            mip=self.levels[channel][resolution];scale=info['scales'][mip]
            grid=StreamingGrid({'source':descriptor['source'],'info':info,'mip':mip},
                  scale['resolution'],scale['voxel_offset'],scale['size'],info['data_type'],
                  lambda lo,hi:self.loader(channel,mip,lo,hi),self.cache,
                  chunk_shape=(128,128,4),fetch_pool=self.fetch_pool,on_labels=getattr(self.ownership,'observe',self.ownership.ensure) if channel=='segmentation' else None)
            self.grids[key]=grid
        return self.grids[key]

    def resolution(self, view):
        requested=view.em_resolution_nm
        if requested is not None:
            options=[r for r in self.shared if r[0]==requested]
            if not options:raise ValueError('Requested resolution is not available for both EM and segmentation')
            return options[0]
        # Automatic level of detail prevents a zoomed-out screen from fetching
        # native-resolution chunks across a millimetre-wide field of view.
        candidates=[r for r in self.shared if r[0]<=view.pixel_size_nm]
        return candidates[-1] if candidates else self.native

    def sampling_grids(self, view):
        resolution=self.resolution(view)
        em=self.level('em',resolution);labels=self.level('segmentation',resolution)
        phase=(em.origin-labels.origin)/em.resolution
        if not np.allclose(phase,np.rint(phase),rtol=0,atol=1e-6):
            raise ValueError('Source voxel grids have different physical alignment')
        return em,labels

    def sampling_metadata(self, view):
        em,labels=self.sampling_grids(view)
        options=[r[0] for r in self.shared]
        return ({'resolution_nm':em.resolution.tolist(),'available_xy_nm':options,
                 'backend':'streamed-source','sampling':'trilinear',
                 'zoom_changes_resolution':view.em_resolution_nm is None,
                 'resolution_mode':'automatic' if view.em_resolution_nm is None else 'explicit'},
                {'resolution_nm':labels.resolution.tolist(),'available_xy_nm':options,
                 'sampling':'nearest','matched_to_em':True})
