"""Complete historical graphs fetched in bounded spatial pieces.

The server's default subgraph response drops edges crossing its query box.
Request external edges explicitly, then retain only edges whose two endpoints
belong to the full historical root. This also preserves long-range manual links.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import uuid
import numpy as np


def boxes(chunks, limit=128):
    """Disjoint boxes covering every occupied chunk, each below the server cap."""
    chunks = np.unique(np.asarray(chunks, dtype=np.int64), axis=0)
    if chunks.ndim != 2 or chunks.shape[1] != 3 or not len(chunks):
        raise ValueError('Missing source graph chunk coordinates')
    if len(chunks) <= limit:
        return [(chunks.min(axis=0), chunks.max(axis=0) + 1)]
    axis = int(np.argmax(np.ptp(chunks, axis=0)))
    values = np.unique(chunks[:, axis])
    pivot = values[len(values)//2]
    left = chunks[:, axis] < pivot
    return boxes(chunks[left], limit) + boxes(chunks[~left], limit)


def validate(edges, weights):
    edges = np.asarray(edges, dtype=np.uint64)
    if edges.size == 0:
        edges = edges.reshape(0, 2)
    weights = np.asarray(weights, dtype=float)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError('Source returned malformed adjacency')
    if weights.shape != (len(edges),) or np.any(np.isnan(weights)) or np.any(weights < 0):
        raise ValueError('Source returned invalid affinity capacities')
    return edges, weights


def request(client, root, bounds):
    # caveclient.get_subgraph does not expose only_internal_edges. Use its
    # authenticated session and endpoint mapping, without changing that session.
    mapping = dict(client.default_url_mapping, root_id=int(root))
    url = client._endpoints['get_subgraph'].format_map(mapping)
    params = {'bounds': '_'.join(f'{int(a)}-{int(b)}' for a,b in bounds),
              'only_internal_edges': 'false', 'int64_as_str': 'true'}
    try:
        response = client.session.get(url, params=params, timeout=90)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        response = getattr(exc, 'response', None)
        status = getattr(response, 'status_code', None)
        # Do not expose authenticated URLs, request headers, or arbitrary HTML.
        detail = f'HTTP {status}' if status is not None else type(exc).__name__
        if status == 400:
            detail += '; source rejected this bounded graph query'
        raise ValueError(f'Selected graph fetch failed for root {root}: {detail}; bounds={bounds.tolist()}') from exc
    return validate(data['nodes'], data['affinities'])


def complete(source, root):
    """Yield bounded edge batches; failed batches never create a usable graph."""
    members = np.asarray(source.members(root), dtype=np.uint64)
    if len(members) < 2:
        return
    volume = source.volumes['segmentation']
    meta = volume.meta
    # Leaf IDs encode their original atomic chunk. No representative-coordinate
    # approximation or expensive per-leaf spatial lookup is needed.
    chunks = [meta.decode_chunk_position(int(a)) for a in members]
    chunk_size = np.asarray(meta.graph_chunk_size, dtype=np.int64)
    offset = np.asarray(meta.voxel_offset(meta.watershed_mip), dtype=np.int64) if meta.chunks_start_at_voxel_offset else np.zeros(3,dtype=np.int64)
    query_boxes = [np.stack((lo*chunk_size+offset,hi*chunk_size+offset),axis=1)
                   for lo,hi in boxes(chunks)]
    directory = Path(source.directory)/'graph-external-v1'
    directory.mkdir(exist_ok=True)

    def fetch(bounds):
        key = hashlib.sha256(json.dumps([int(root),bounds.tolist()]).encode()).hexdigest()
        path = directory/(key+'.npz')
        if path.exists():
            try:
                with np.load(path,allow_pickle=False) as data:
                    return validate(data['edges'],data['weights'])
            except (ValueError,KeyError,OSError):
                path.unlink(missing_ok=True)
        edges,weights = request(source.client.chunkedgraph,root,bounds)
        # External edges include other roots: never let these into our partition.
        mask = np.isin(edges[:,0],members) & np.isin(edges[:,1],members)
        edges,weights = edges[mask],weights[mask]
        temporary = path.with_suffix('.'+uuid.uuid4().hex+'.partial')
        try:
            with temporary.open('wb') as f:
                np.savez_compressed(f,edges=edges,weights=weights)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return edges,weights

    # At most four in-flight responses; do not accumulate the full graph twice.
    with ThreadPoolExecutor(max_workers=4,thread_name_prefix='source-graph') as pool:
        for start in range(0,len(query_boxes),4):
            yield from pool.map(fetch,query_boxes[start:start+4])
