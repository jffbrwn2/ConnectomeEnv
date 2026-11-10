"""Matched physical sampling of EM and immutable segmentation labels."""
import numpy as np
from .multiscale_em import sampling_metadata


def grids(dataset, view):
    if hasattr(dataset, "sampling_grids"):
        return dataset.sampling_grids(view)
    sampling_metadata(dataset.em, view)
    em = dataset.em.for_view(view) if hasattr(dataset.em, 'for_view') else dataset.em
    candidates = getattr(dataset, 'segmentation_levels', [dataset.atoms])
    atoms = next((g for g in candidates if np.array_equal(g.resolution, em.resolution)), None)
    if atoms is None:
        raise ValueError('No matching segmentation at EM resolution '
                         f'{em.resolution.tolist()} nm; prepare a matched dataset or select a shared resolution')
    # Equal voxel spacing is insufficient if the grids have different phases.
    phase = (atoms.origin - getattr(em, 'origin', np.zeros(3))) / em.resolution
    if not np.allclose(phase, np.rint(phase), rtol=0, atol=1e-6):
        raise ValueError('EM and segmentation voxel grids are not aligned')
    return em, atoms


def metadata(dataset, view):
    if hasattr(dataset, "sampling_metadata"):
        return dataset.sampling_metadata(view)
    em, atoms = grids(dataset, view)
    result = sampling_metadata(dataset.em, view)
    available = []
    from dataclasses import replace
    for xy in result['available_xy_nm']:
        try:
            grids(dataset, replace(view, em_resolution_nm=xy))
        except ValueError:
            continue
        available.append(xy)
    result['available_xy_nm'] = available
    return result, {'resolution_nm': atoms.resolution.tolist(),
                    'sampling': 'nearest', 'matched_to_em': True,
                    'available_xy_nm': available}


def load_levels(dataset, config, base):
    """Load frozen local source levels; never interpolate or infer label IDs."""
    from .volume import Grid
    from .storage import data_path
    levels = [dataset.atoms]
    for c in config.get('segmentation_pyramid', []):
        path = data_path(c['path'], base)
        if hasattr(dataset, 'identity_kind'):
            from .evaluate import file_hash
            if not c.get('sha256') or file_hash(path) != c['sha256']:
                raise ValueError('Indexed segmentation level requires a matching frozen SHA256')
        g = Grid(np.load(path, mmap_mode='r', allow_pickle=False),
                 c['resolution_nm'], c['origin_nm'])
        if g.data.dtype.kind not in 'iu' or np.any(g.data < 0):
            raise ValueError('Segmentation levels require nonnegative integer labels')
        if not np.array_equal(g.origin, dataset.atoms.origin) or not np.array_equal(g.upper, dataset.atoms.upper):
            raise ValueError('Segmentation levels must cover the same physical crop')
        if any(np.array_equal(g.resolution, other.resolution) for other in levels):
            raise ValueError('Duplicate segmentation resolution')
        # An atom absent from the editable universe cannot silently be assigned to
        # a current root or to its nearest coarse neighbor.
        known = dataset.ids if hasattr(dataset, 'ids') else dataset.native
        labels = np.unique(g.data)
        if np.any(~np.isin(labels[labels != 0], known)):
            raise ValueError('Segmentation level contains atoms missing from the frozen partition')
        levels.append(g)
    dataset.segmentation_levels = levels
