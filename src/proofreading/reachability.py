"""Evaluator-only crop reachability in the environment's immutable-atom graph."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def reachable_nodes(nodes, edges, seeds):
    nodes=np.asarray(nodes,dtype=np.int64)
    if nodes.ndim!=1 or len(np.unique(nodes))!=len(nodes) or np.any(nodes<=0):
        raise ValueError('Nodes must be unique positive IDs')
    nodes=np.sort(nodes);seeds=np.asarray(seeds,dtype=np.int64)
    if not len(seeds) or not np.isin(seeds,nodes).all():raise ValueError('Seeds must belong to target graph')
    edges=np.asarray(edges,dtype=np.int64).reshape(-1,2)
    positions=np.searchsorted(nodes,edges)
    valid=(positions<len(nodes)).all(axis=1)
    valid[valid]&=(nodes[positions[valid]]==edges[valid]).all(axis=1)
    positions=positions[valid]
    graph=coo_matrix((np.ones(len(positions),dtype=np.uint8),(positions[:,0],positions[:,1])),shape=(len(nodes),len(nodes))).tocsr()
    _,labels=connected_components(graph,directed=False)
    seed_labels=labels[np.searchsorted(nodes,seeds)]
    return nodes[np.isin(labels,seed_labels)]
