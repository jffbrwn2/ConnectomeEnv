import networkx as nx
import pytest

from proofreading.seeded_split import split


def graph(edges, nodes=()):
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_weighted_edges_from(edges, weight='capacity')
    return g


def test_connecting_marks_prevents_cutting_the_middle_out_of_a_process():
    g = graph([(1, 2, 1), (2, 3, 1), (2, 4, 10)])
    old, _ = split(g, {1, 3}, {4}, 'min_cut')
    assert old == [{1, 3}, {2, 4}]
    new, diagnostics = split(g, {1, 3}, {4})
    assert new == [{1, 2, 3}, {4}]
    assert diagnostics['protected_atom_counts'] == [3, 1]
    assert set(g) == {1, 2, 3, 4}  # terminals must not leak into source graph
    assert g[2][4]['capacity'] == 10


def test_protected_edges_are_indivisible_during_path_search_and_cut():
    g = graph([(1, 2, float('inf')), (2, 3, 2), (3, 4, 1)])
    with pytest.raises(ValueError, match='protected infinite'):
        split(g, {1}, {2})
    sides, _ = split(g, {1, 3}, {4})
    assert sides == [{1, 2, 3}, {4}]


def test_crossed_marks_report_conflict_instead_of_silently_severing_paths():
    g = graph([(1, 2, 1), (2, 3, 1), (3, 4, 1)])
    with pytest.raises(ValueError, match='Cannot connect'):
        split(g, {1, 3}, {2, 4})
    sides, _ = split(g, {1, 3}, {2, 4}, 'min_cut')
    assert sides == [{1, 3}, {2, 4}]


def test_side_names_do_not_change_cut():
    g = graph([(1, 2, 2), (2, 3, 2), (2, 4, 3), (4, 5, 2)])
    forward, _ = split(g, {1, 3}, {4, 5})
    reverse, _ = split(g, {4, 5}, {1, 3})
    assert forward == reverse[::-1]


def test_disconnected_seeded_pieces_and_unseeded_atoms_are_preserved():
    g = graph([(1, 2, 1)], nodes=[3, 4])
    sides, _ = split(g, {1, 3}, {2})
    assert {1, 3} <= sides[0] and 2 in sides[1]
    assert sides[0] | sides[1] == set(g)
    assert not sides[0] & sides[1]


def test_zero_capacity_contact_does_not_assert_continuity():
    g = graph([(1, 2, 0), (2, 3, 1)])
    with pytest.raises(ValueError, match='Cannot connect'):
        split(g, {1, 2}, {3})


@pytest.mark.parametrize('source,sink', [([], [2]), ([1], [1]), ([1], [8])])
def test_invalid_marks(source, sink):
    with pytest.raises(ValueError, match='distinct atoms'):
        split(graph([(1, 2, 1)]), source, sink)


def test_invalid_algorithm():
    with pytest.raises(ValueError, match='Unknown'):
        split(graph([(1, 2, 1)]), [1], [2], 'guess')


def test_session_preview_commit_undo_and_replay_with_connected_marks(tmp_path):
    import json
    import numpy as np
    from proofreading.volume import Dataset, Grid
    from proofreading.session import Session
    from proofreading.replay import replay
    atoms = np.zeros((12, 12, 12), dtype=np.uint64)
    atoms[2:4, 2:6, 2:6] = 1
    atoms[4:6, 2:6, 2:6] = 2
    atoms[6:8, 2:6, 2:6] = 3
    atoms[4:6, 6:8, 2:6] = 4
    d = Dataset(Grid(atoms, [1, 1, 1], [0, 0, 0]),
                Grid(np.full(atoms.shape, 100, np.uint8), [1, 1, 1], [0, 0, 0]),
                {i: 10 for i in range(1, 5)}, [(1, 2, 1), (2, 3, 1), (2, 4, 10)])
    session = Session(d, tmp_path/'session')
    args = dict(segment='10', source_nm=[[2.5, 2.5, 2.5], [6.5, 2.5, 2.5]],
                sink_nm=[[4.5, 6.5, 2.5]])
    response = session.execute('preview_split', args)
    assert response['ok'], response
    result = response['result']
    assert result['split_algorithm'] == 'connected_min_cut'
    assert result['protected_atom_counts'] == [3, 1]
    assert session.partition[1] == session.partition[4]  # preview does not edit
    assert session.execute('commit', {'preview_id': result['preview_id']})['ok']
    assert session.partition[1] == session.partition[2] == session.partition[3]
    assert session.partition[1] != session.partition[4]
    assert session.execute('undo_edit', {})['ok']
    assert len(set(session.partition.values())) == 1
    assert session.execute('redo_edit', {})['ok']
    session.execute('finish', {'summary': 'test'})
    assert json.loads((session.output/'session.json').read_text())['split_algorithm'] == 'connected_min_cut'
    assert replay(d, session.output, tmp_path/'replay')['partition_equal']

    # Historical sessions without the algorithm field still use the old cut.
    legacy = Session(d, tmp_path/'legacy', split_algorithm='min_cut')
    preview = legacy.execute('preview_split', args)['result']
    legacy.execute('commit', {'preview_id': preview['preview_id']})
    legacy.execute('finish', {'summary': 'legacy'})
    meta_path = legacy.output/'session.json'
    meta = json.loads(meta_path.read_text())
    del meta['split_algorithm']
    meta_path.write_text(json.dumps(meta))
    assert replay(d, legacy.output, tmp_path/'legacy-replay')['partition_equal']
