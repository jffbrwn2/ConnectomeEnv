"""Seeded cuts with optional explicit same-side continuity constraints.

No geometry, reference labels, or scoring data are used here. Edge capacities
are the source segmentation's affinities, not probabilities of correctness.
"""
import math

import networkx as nx


def _contract(graph):
    """Protected edges are indivisible, including during path construction."""
    union = nx.utils.UnionFind(graph)
    for a, b, data in graph.edges(data=True):
        weight = float(data['capacity'])
        if math.isnan(weight) or weight < 0:
            raise ValueError('Split graph has invalid capacities')
        if math.isinf(weight):
            union.union(a, b)
    members = {}
    for atom in sorted(graph):
        members.setdefault(union[atom], set()).add(atom)
    owner = {a: min(group) for group in members.values() for a in group}
    reduced = nx.Graph()
    reduced.add_nodes_from(sorted(set(owner.values())))
    for a, b, data in graph.edges(data=True):
        u, v = owner[a], owner[b]
        if u != v:
            previous = reduced.get_edge_data(u, v, {}).get('capacity', 0.)
            reduced.add_edge(u, v, capacity=previous + float(data['capacity']))
    return reduced, owner


def _paths(graph, seeds, forbidden):
    """Join marks along high-affinity paths, without crossing opposing marks."""
    tree = {min(seeds)}
    cost = 0.
    allowed = nx.subgraph_view(graph, filter_node=lambda a: a not in forbidden)

    def resistance(a, b, data):
        capacity = data['capacity']
        # Zero-affinity contacts do not establish continuity.
        return 1. / capacity if capacity > 0 else None

    for target in sorted(seeds - tree):
        if target in tree:
            continue
        distance, path = nx.multi_source_dijkstra(
            allowed, sorted(tree), target=target, weight=resistance)
        tree.update(path)
        cost += distance
    return tree, cost


def continuity_seeds(graph, source, sink):
    reduced, owner = _contract(graph)
    seeds = [{owner[a] for a in side} for side in (source, sink)]
    if seeds[0] & seeds[1]:
        raise ValueError('Split seeds are connected by protected infinite-capacity source edges')
    protected = [set(), set()]
    # Marks in already disconnected components retain their explicit side.
    # Do not fabricate paths between them.
    for component in nx.connected_components(reduced):
        local = [side & component for side in seeds]
        if not any(local):
            continue
        subgraph = reduced.subgraph(component)
        candidates = []
        for order in ((0, 1), (1, 0)):
            groups = [set(side) for side in local]
            cost = 0.
            try:
                for side in order:
                    if groups[side]:
                        groups[side], distance = _paths(subgraph, local[side], groups[1-side])
                        cost += distance
            except nx.NetworkXNoPath:
                continue
            # Tie-breaking is independent of source/sink naming.
            key = tuple(sorted(tuple(sorted(group)) for group in groups))
            candidates.append((cost, key, groups))
        if not candidates:
            raise ValueError(
                'Cannot connect the marks on each side without crossing the other side. '
                'Inspect the junction from another angle and revise the seed groups, '
                'or use algorithm="min_cut" if disconnected same-side pieces are intended.')
        groups = min(candidates, key=lambda item: (item[0], item[1]))[2]
        for side in (0, 1):
            protected[side].update(groups[side])
    expanded = [{a for a in graph if owner[a] in side} for side in protected]
    return expanded


def split(graph, source, sink, algorithm='connected_min_cut'):
    """Return a complete two-side partition and public, reference-free diagnostics."""
    if algorithm not in {'min_cut', 'connected_min_cut'}:
        raise ValueError('Unknown split algorithm')
    source, sink = set(source), set(sink)
    if not source or not sink or source & sink or not (source | sink) <= graph.nodes:
        raise ValueError('Seeds must lie in distinct atoms of this segment')
    expanded = (continuity_seeds(graph, source, sink)
                if algorithm == 'connected_min_cut' else [source, sink])
    working = graph.copy()
    start, end = object(), object()
    for terminal, side in zip((start, end), expanded):
        for atom in sorted(side):
            working.add_edge(terminal, atom, capacity=float('inf'))
    try:
        cost, (left, right) = nx.minimum_cut(working, start, end, capacity='capacity')
    except nx.NetworkXUnbounded as exc:
        raise ValueError('Split seeds are connected by protected infinite-capacity source edges') from exc
    sides = [left - {start, end}, right - {start, end}]
    return sides, {
        'split_algorithm': algorithm,
        'seed_atom_counts': [len(source), len(sink)],
        'protected_atom_counts': list(map(len, expanded)),
        'cut_capacity': float(cost),
        'continuity_note': (
            'Same-side marks are joined by affinity paths that the cut must preserve. '
            'These paths are inferred, not anatomical proof; inspect both preview pieces.'
            if algorithm == 'connected_min_cut' else
            'Only the marked atoms are constrained; unmarked paths between them may be cut.'),
    }
