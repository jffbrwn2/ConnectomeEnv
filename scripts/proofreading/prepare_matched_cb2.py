"""Prepare new native-resolution CB2 candidates from frozen historical cases.

Downloads native supervoxels, never current root segmentation. Ownership is
queried at the original operation timestamps. Original data are never modified.
Use freeze_cb2_practice.py afterwards to create new scoring contracts.
"""
import argparse
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import numpy as np
try:
    from .prepare_cb2_practice import reference_labels, composition
except ImportError:
    from prepare_cb2_practice import reference_labels, composition



def coarser_labels(native, ratio):
    """Nearest categorical voxel at each aligned coarse voxel center."""
    if ratio not in (2,4) or any(n % ratio for n in native.shape[:2]):
        raise ValueError('Native XY shape must align with coarse resolution')
    return native[ratio//2::ratio,ratio//2::ratio,:]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidates',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--operation',action='append')
    p.add_argument('--native-cache',type=Path,help='Previously completed native downloads, laid out as OP/public/atoms.npy')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    from caveclient import CAVEclient
    from cloudvolume import Bbox
    client=CAVEclient('minnie65_public')
    volume=client.info.segmentation_cloudvolume(agglomerate=False,progress=False,parallel=False)
    reports=[]
    eligible={p.name for p in args.candidates.iterdir() if (p/'evaluator/audit.json').exists()}
    if args.operation and not set(args.operation)<=eligible:raise ValueError('Unknown requested operation')
    if not eligible:raise ValueError('No historical candidates found')
    for source in sorted(args.candidates.iterdir()):
        audit_path=source/'evaluator/audit.json'
        if not audit_path.exists() or (args.operation and source.name not in args.operation):continue
        audit=json.loads(audit_path.read_text());old=json.loads((source/'public/dataset.json').read_text())
        dest=args.output/source.name;public=dest/'public';private=dest/'evaluator'
        public.mkdir(parents=True);private.mkdir()
        origin=np.asarray(old['atoms']['origin_nm']);coarse=np.asarray(old['atoms']['resolution_nm'])
        original=np.load(source/'public/atoms.npy',mmap_mode='r')
        upper=origin+coarse*np.asarray(original.shape)
        levels=[]
        for xy in (8,16,32):
            mip=next(i for i,s in enumerate(volume.info['scales']) if s['resolution']==[xy,xy,40])
            resolution=np.array([xy,xy,40]);lo=origin/resolution;hi=upper/resolution
            if not np.array_equal(lo,np.rint(lo)) or not np.array_equal(hi,np.rint(hi)):
                raise ValueError('Crop not aligned with source segmentation')
            lo=lo.astype(int);hi=hi.astype(int);filename='atoms.npy' if xy==8 else f'atoms-{xy}.npy'
            cached=args.native_cache/source.name/'public/atoms.npy' if args.native_cache else None
            if xy != 8:
                # Build every coarse label grid from this exact native crop. A
                # server's independently downsampled boundary voxel can otherwise
                # contain an atom lying just outside the crop at native resolution.
                native=np.load(public/'atoms.npy',mmap_mode='r')
                ratio=xy//8
                np.save(public/filename,coarser_labels(native,ratio))
            elif cached is not None and cached.exists():
                test=np.load(cached,mmap_mode='r')
                if test.shape!=tuple(hi-lo) or test.dtype!=np.uint64:raise ValueError('Native cache shape/dtype mismatch')
                os.link(cached,public/filename)
            else:
                array=np.lib.format.open_memmap(public/(filename+'.partial'),mode='w+',dtype=np.uint64,shape=tuple(hi-lo))
                # Bounded downloads; labels stay uint64 throughout.
                for z in range(lo[2],hi[2],8):
                    end=min(z+8,hi[2]);a=lo.copy();b=hi.copy();a[2]=z;b[2]=end
                    raw=np.asarray(volume.download(Bbox(a,b),mip=mip,agglomerate=False,coord_resolution=volume.meta.resolution(mip)))
                    if raw.ndim==4:raw=raw[...,0]
                    array[:,:,z-lo[2]:end-lo[2]]=raw
                array.flush();del array
                (public/(filename+'.partial')).replace(public/filename)
            levels.append({'path':filename,'resolution_nm':resolution.tolist(),'origin_nm':origin.tolist()})
        cube=np.load(public/'atoms.npy',mmap_mode='r')
        ids,counts=np.unique(cube,return_counts=True);foreground=ids!=0;atoms=ids[foreground]
        dropped=np.setdiff1d(np.unique(original),ids)
        dropped=dropped[dropped!=0]
        timestamp=datetime.fromisoformat(audit['timestamp'])
        partitions=[]
        for label,when in [('before',timestamp-timedelta(microseconds=1)),('after',timestamp+timedelta(microseconds=1))]:
            owners=np.empty(len(atoms),np.uint64)
            for start in range(0,len(atoms),50000):
                owners[start:start+50000]=client.chunkedgraph.get_roots(atoms[start:start+50000],timestamp=when)
            if (owners==0).any():raise ValueError('Missing historical ownership; refusing guessed labels')
            with np.load(source/'evaluator'/f'{label}.npz') as previous:
                shared=np.isin(previous['atoms'],atoms)
                pos=np.searchsorted(atoms,previous['atoms'][shared])
                if not np.array_equal(owners[pos],previous['segments'][shared]):raise ValueError('Historical ownership drift')
            np.savez(private/f'{label}.npz',atoms=atoms,segments=owners);partitions.append(owners)
        before,after=partitions
        reference=Path(audit['reference_spec']).parent
        known=np.load(reference/'atoms.npy',mmap_mode='r');roots=np.load(reference/'reference.npy',mmap_mode='r')
        ref=reference_labels(atoms,known,roots)
        lookup=np.zeros(len(ids),np.uint64);lookup[foreground]=ref
        truth=np.lib.format.open_memmap(private/'reference.npy',mode='w+',dtype=np.uint64,shape=cube.shape)
        for z in range(cube.shape[2]):truth[:,:,z]=lookup[np.searchsorted(ids,cube[:,:,z])]
        truth.flush();del truth
        os.link(source/'public/em.npy',public/'em.npy')
        np.savez(public/'partition.npz',atoms=atoms,segments=before)
        config={'name':'matched-resolution-practice','partition':'partition.npz','atoms':levels[0],
                'em':old['em'],'segmentation_pyramid':levels[1:]}
        (public/'dataset.json').write_text(json.dumps(config,indent=2)+'\n')
        before_ids=list(map(int,audit['before_certified_composition']));after_ids=list(map(int,audit['after_certified_composition']))
        affected=np.isin(before,before_ids)
        audit.update(before_certified_composition=composition(before,ref,counts[foreground],before_ids),
                     after_certified_composition=composition(after,ref,counts[foreground],after_ids),
                     unexpected_after_roots=np.unique(after[affected & ~np.isin(after,after_ids)]).tolist(),
                     foreground_atoms=len(atoms),foreground_voxels=int(counts[foreground].sum()),
                     certified_voxels=int(counts[foreground][ref!=0].sum()),
                     matched_resolution={'source':client.info.segmentation_source(),'xy_nm':[8,16,32],
                                         'source_candidates':str(source),'new_scoring_grid_nm':[8,8,40],
                                         'coarse_levels':'nearest native voxel at coarse voxel center',
                                         'coarse_atoms_absent_from_native_crop':list(map(str,dropped)),
                                         'historical_ownership_verified':True,'reference_for_new_atoms':'unknown unless present in frozen certified lookup'})
        (private/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');reports.append(audit)
        print(json.dumps({'operation':source.name,'atoms':len(atoms),'shape':list(cube.shape),'status':'prepared'}),flush=True)
    (args.output/'audits.json').write_text(json.dumps(reports,indent=2)+'\n')


if __name__=='__main__':main()
