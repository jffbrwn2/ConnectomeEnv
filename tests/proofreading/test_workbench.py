import json
import numpy as np
import pytest
from proofreading.volume import Dataset, Grid
from proofreading.session import Session
from proofreading.evaluate import metrics


@pytest.fixture
def dataset():
    a = np.zeros((20, 20, 20), np.uint64)
    a[2:9, 2:12, 2:12] = 101
    a[9:16, 2:12, 2:12] = 102
    a[2:9, 12:18, 2:12] = 103
    return Dataset(
        Grid(a, [2, 3, 4], [100, 200, 300]),
        Grid(np.full(a.shape, 100, dtype=np.uint8), [2, 3, 4], [100, 200, 300]),
        {101: 9007199254740993, 102: 12, 103: 13},
    )


@pytest.fixture
def session(dataset, tmp_path):
    return Session(dataset, tmp_path / "session")


def call(s, name, **args):
    r = s.execute(name, args)
    assert r["ok"], r
    return r["result"]


def test_oblique_pixel_world_and_large_id(session):
    q = np.sqrt(0.5)
    v = call(
        session,
        "set_view",
        center_nm=[110, 215, 320],
        right=[q, q, 0],
        down=[0, 0, 1],
        width=16,
        height=16,
        pixel_size_nm=1.0,
    )
    hit = call(session, "click", view_id=v["view_id"], x=7.5, y=7.5, mode="pick")
    assert hit["segment"] == "9007199254740993"
    np.testing.assert_allclose(hit["xyz_nm"], [110, 215, 320])
    p = (
        np.array(v["transform"]["pixel_origin_nm"])
        + 7.5 * np.array(v["transform"]["pixel_right_nm"])
        + 7.5 * np.array(v["transform"]["pixel_down_nm"])
    )
    np.testing.assert_allclose(p, hit["xyz_nm"])


def test_different_em_grid_alignment():
    x, y, z = np.indices((40, 40, 40))
    em = Grid(
        (x + 0.5) * 1 + (y + 0.5) * 1.5 + (z + 0.5) * 2, [1, 1.5, 2], [100, 200, 300]
    )
    p = np.array([110, 215, 320.0])
    value, valid = em.linear(p.reshape(1, 1, 3))
    assert valid.item()
    assert value.item() == pytest.approx(45.0)


def test_selection_never_edits(session):
    old = dict(session.partition)
    call(session, "select", segments=["12"], mode="hide")
    call(session, "select", segments=["13"], mode="replace")
    assert session.partition == old and session.revision == 0


def test_merge_undo_redo_and_stale_click(session):
    old = dict(session.partition)
    image = call(session, "look")
    p = call(session, "preview_merge", segments=["12", "13"])
    assert session.partition == old
    result = call(session, "commit", preview_id=p["preview_id"])
    merged = dict(session.partition)
    assert len(result["children"]) == 1
    assert not session.execute("click", {"view_id": image["view_id"], "x": 2, "y": 2})[
        "ok"
    ]
    assert not session.execute("commit", {"preview_id": p["preview_id"]})["ok"]
    call(session, "undo_edit")
    assert session.partition == old
    call(session, "redo_edit")
    assert session.partition == merged


def test_split_all_children_and_atom_conservation(session):
    p = call(session, "preview_merge", segments=["9007199254740993", "12", "13"])
    sid = call(session, "commit", preview_id=p["preview_id"])["children"][0]
    p = call(session, "preview_split", segment=sid, groups=[["101"], ["102"], ["103"]])
    result = call(session, "commit", preview_id=p["preview_id"])
    assert len(result["children"]) == 3
    assert set(session.partition) == {101, 102, 103}
    assert len(set(session.partition.values())) == 3


def test_seeded_mincut(session):
    p = call(session, "preview_merge", segments=["9007199254740993", "12"])
    sid = call(session, "commit", preview_id=p["preview_id"])["children"][0]
    p = call(
        session,
        "preview_split",
        segment=sid,
        source_nm=[[110, 215, 320]],
        sink_nm=[[124, 215, 320]],
    )
    assert sorted(p["child_atom_counts"]) == [1, 1]


@pytest.mark.parametrize(
    "groups", [[["101"], ["101", "102"]], [["101"], []], [["101"], ["102", "999"]]]
)
def test_invalid_split_is_atomic(session, groups):
    before = dict(session.partition)
    assert not session.execute(
        "preview_split", {"segment": "9007199254740993", "groups": groups}
    )["ok"]
    assert session.partition == before and not session.undo_stack


def test_invalid_view_does_not_move_camera(session):
    old = session.state()
    assert not session.execute("set_view", {"right": [1, 0, 0], "down": [1, 0, 0]})[
        "ok"
    ]
    assert session.state() == old


def test_score_false_merges_false_splits_and_unknown_scope():
    atoms = np.array([1, 2, 3, 4, 5], dtype=np.uint64).reshape(5, 1, 1)
    truth = np.array([1, 1, 2, 2, 0], dtype=np.uint64).reshape(5, 1, 1)
    scope = np.ones(atoms.shape, bool)
    correct = {1: 10, 2: 10, 3: 20, 4: 20, 5: 99}
    m = metrics(atoms, correct, truth, scope)
    assert m["vi_total_bits"] == 0
    assert m["scored_voxels"] == 4 and m["unknown_reference_voxels_in_scope"] == 1
    m = metrics(atoms, {i: 10 for i in range(1, 6)}, truth, scope)
    assert m["vi_merge_bits"] == pytest.approx(1) and m["vi_split_bits"] == 0
    m = metrics(atoms, {i: i for i in range(1, 6)}, truth, scope)
    assert m["vi_split_bits"] == pytest.approx(1) and m["vi_merge_bits"] == 0
    scope[2:] = False
    assert (
        metrics(atoms, {i: 10 for i in range(1, 6)}, truth, scope)["vi_total_bits"] == 0
    )


def test_no_reference_tool_or_reference_in_observations(session):
    from proofreading.tools import schemas

    assert all("evaluate" != s["name"] for s in schemas())
    assert not session.execute("evaluate", {})["ok"]
    state = call(session, "state")
    assert "reference" not in state and "reward" not in state


def test_replay_exact(dataset, session, tmp_path):
    from proofreading.replay import replay

    call(session, "look")
    p = call(session, "preview_merge", segments=["12", "13"])
    call(session, "commit", preview_id=p["preview_id"])
    call(session, "undo_edit")
    call(session, "redo_edit")
    result = replay(dataset, session.output, tmp_path / "replayed")
    assert result["partition_equal"] and result["exact_images"] == 10


def test_http_reference_is_inaccessible(session):
    import threading
    from urllib.request import urlopen, Request
    from urllib.error import HTTPError
    from proofreading.server import make_server

    server = make_server(session, 0)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        assert b"Proofreading workbench" in urlopen(base).read()
        with pytest.raises(HTTPError) as e:
            urlopen(base + "/api/state")
        assert e.value.code == 401
        for route in [
            "/evaluation/reference.npy",
            "/../session.json",
            "/frames/../partition.npz",
        ]:
            with pytest.raises(HTTPError) as e:
                urlopen(
                    Request(
                        base + route,
                        headers={"Authorization": "Bearer " + server.agent_token},
                    )
                )
            assert e.value.code == 404
        payload = json.dumps({"name": "look", "arguments": {}}).encode()
        r = json.load(
            urlopen(
                Request(
                    base + "/api/tool",
                    payload,
                    {
                        "Authorization": "Bearer " + server.agent_token,
                        "Content-Type": "application/json",
                    },
                )
            )
        )
        assert r["ok"]
    finally:
        server.shutdown()
        server.server_close()
        t.join()


@pytest.mark.parametrize("provider", ["responses", "anthropic"])
def test_model_image_tool_loop_and_budget(session, provider):
    from proofreading.harness import run

    from proofreading.objectives import AXON_EXTENSION
    calls = []

    def transport(endpoint, payload, headers):
        calls.append(payload)
        assert AXON_EXTENSION in payload["instructions" if provider == "responses" else "system"]
        if len(calls) == 1:
            blocks = payload["input"][0]["content"] if provider == "responses" else payload["messages"][0]["content"]
            image_type = "input_image" if provider == "responses" else "image"
            assert sum(block["type"] == image_type for block in blocks) == 2
            initial_steps = json.loads((session.output / "initial-observations.json").read_text())
            assert [e["tool"] for e in session.events if e["step"] in initial_steps] == ["look"]
            assert session.events[-3]["arguments"] == session.scene
            if provider == "responses":
                assert payload["input"][0]["content"][1]["type"] == "input_image"
                return {
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "a",
                            "name": "look",
                            "arguments": "{}",
                        }
                    ],
                    "usage": {"input_tokens": 100},
                }
            assert payload["messages"][0]["content"][1]["type"] == "image"
            return {
                "content": [
                    {"type": "tool_use", "id": "a", "name": "look", "input": {}}
                ]
            }
        if provider == "responses":
            assert payload["input"][-1]["output"][1]["type"] == "input_image"
            return {
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "b",
                        "name": "finish",
                        "arguments": '{"summary":"done"}',
                    }
                ]
            }
        assert payload["messages"][-1]["content"][0]["content"][1]["type"] == "image"
        return {
            "content": [
                {
                    "type": "tool_use",
                    "id": "b",
                    "name": "finish",
                    "input": {"summary": "done"},
                }
            ]
        }

    report = run(
        session,
        "explicit-test-model",
        provider,
        api_key="test-only",
        transport=transport,
        objective=AXON_EXTENSION,
    )
    assert report["status"] == "finished" and report["actions"] == 2
    assert "test-only" not in (session.output / "harness.json").read_text()

    assert report["objective"] == AXON_EXTENSION
    assert AXON_EXTENSION in (session.output / "system-prompt.txt").read_text()


def test_model_action_budget(session):
    from proofreading.harness import run

    def transport(*args):
        return {
            "output": [
                {
                    "type": "function_call",
                    "call_id": "a",
                    "name": "look",
                    "arguments": "{}",
                }
            ]
        }

    report = run(session, "test", max_actions=1, api_key="test", transport=transport)
    assert report["actions"] == 1
    assert report["status"] == "action_budget"


def test_invalid_slab_rolls_back_camera(session):
    old = session.state()
    assert not session.execute("set_view", {"slab_nm": 100000, "projection": "mean"})[
        "ok"
    ]
    assert session.state() == old


def test_schema_rejects_rounded_numeric_ids(session):
    old = dict(session.partition)
    assert not session.execute("preview_merge", {"segments": [12, 13]})["ok"]
    assert session.partition == old


def test_colors_preserve_large_id_differences():
    from proofreading.view import color

    assert not np.allclose(color(864691135000000001), color(864691135000000002))


def test_native_neuroglancer_sync(tmp_path):
    pytest.importorskip("neuroglancer")
    from proofreading.demo import create
    from proofreading.neuroglancer_bridge import NeuroglancerBridge

    s = Session(Dataset.load(create(tmp_path / "data")), tmp_path / "episode")
    bridge = NeuroglancerBridge(s)
    try:
        call(s, "select", segments=["11"])
        bridge.sync()
        assert set(bridge.viewer.state.layers["segments"].segments) == {11}
        p = call(s, "preview_merge", segments=["11", "12"])
        sid = call(s, "commit", preview_id=p["preview_id"])["children"][0]
        bridge.sync()
        assert bridge.revision == 1
        assert set(bridge.viewer.state.layers["segments"].segments) == {int(sid)}
        assert {layer.name for layer in bridge.viewer.state.layers} == {
            "em",
            "segments",
        }
    finally:
        bridge.close()


def test_import_world_csr_and_separate_em_grid(tmp_path):
    from proofreading.import_world import import_world

    world = tmp_path / "world"
    world.mkdir()
    csr = world / "root_atoms_csr"
    csr.mkdir()
    seg = np.ones((4, 4, 4), np.uint64)
    seg[2:] = 2
    np.save(world / "seg.npy", seg)
    np.save(world / "em.npy", np.ones((8, 8, 4), np.uint8) * 120)
    np.save(csr / "root_ids.npy", np.array([11, 12, 99]))
    np.save(csr / "indptr.npy", np.array([0, 1, 2, 4]))
    np.save(csr / "atoms.npy", np.array([1, 2, 1, 2]))
    np.save(csr / "initial_roots.npy", np.array([11, 12]))
    (world / "manifest.json").write_text(
        json.dumps(
            {
                "resolution_xyz_nm": [4, 4, 8],
                "bounds_min_xyz": [10, 20, 30],
                "extra": {
                    "em_grid": {
                        "resolution_xyz_nm": [2, 2, 8],
                        "bounds_min_xyz": [20, 40, 30],
                    }
                },
            }
        )
    )
    (world / "reference.json").write_text("DO NOT READ REFERENCE")
    path = import_world(world, tmp_path / "crop", [1, 1, 1, 4, 4, 4])
    d = Dataset.load(path)
    assert d.initial == {1: 11, 2: 12}
    np.testing.assert_equal(d.atoms.origin, [44, 84, 248])
    np.testing.assert_equal(d.em.origin, d.atoms.origin)
    assert d.em.data.shape == (6, 6, 3)
    assert not (path.parent / "reference.json").exists()


def test_frozen_evaluation_rejects_changed_scope_and_late_freeze(dataset, tmp_path):
    from proofreading.evaluate import freeze, evaluate_frozen

    reference = np.array(dataset.atoms.data)
    scope = reference > 0
    np.save(tmp_path / "reference.npy", reference)
    np.save(tmp_path / "scope.npy", scope)
    spec = tmp_path / "spec.json"
    freeze(dataset, tmp_path / "reference.npy", tmp_path / "scope.npy", spec)
    session = Session(dataset, tmp_path / "scored-session")
    assert evaluate_frozen(dataset, session.output, spec)["frozen_before_episode"]
    scope[2, 2, 2] = False
    np.save(tmp_path / "scope.npy", scope)
    with pytest.raises(ValueError, match="changed"):
        evaluate_frozen(dataset, session.output, spec)
    late = tmp_path / "late.json"
    freeze(dataset, tmp_path / "reference.npy", tmp_path / "scope.npy", late)
    with pytest.raises(ValueError, match="after"):
        evaluate_frozen(dataset, session.output, late)


@pytest.mark.parametrize("provider", ["responses", "anthropic"])
def test_provider_over_real_http(session, provider):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from proofreading.harness import run

    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            name = "look" if len(requests) == 1 else "finish"
            args = {} if name == "look" else {"summary": "Local protocol test"}
            if provider == "responses":
                assert self.headers["Authorization"] == "Bearer local-test"
                response = {
                    "output": [
                        {
                            "type": "function_call",
                            "name": name,
                            "arguments": json.dumps(args),
                            "call_id": str(len(requests)),
                        }
                    ]
                }
            else:
                assert self.headers["x-api-key"] == "local-test"
                response = {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": name,
                            "input": args,
                            "id": str(len(requests)),
                        }
                    ]
                }
            data = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        report = run(
            session,
            "protocol-test",
            provider,
            api_key="local-test",
            endpoint=f"http://127.0.0.1:{server.server_port}/api",
            model_options={"temperature": 0.2},
        )
        assert report["status"] == "finished" and len(requests) == 2
        assert requests[0]["temperature"] == 0.2
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_supplied_capacity_graph_controls_split(dataset, tmp_path):
    d = Dataset(
        dataset.atoms,
        dataset.em,
        {101: 1, 102: 1, 103: 1},
        [(101, 102, 1.0), (102, 103, 10.0)],
    )
    s = Session(d, tmp_path / "capacities")
    p = call(
        s,
        "preview_split",
        segment="1",
        source_nm=[[110, 215, 320]],
        sink_nm=[[110, 245, 320]],
    )
    call(s, "commit", preview_id=p["preview_id"])
    assert s.partition[102] == s.partition[103] and s.partition[101] != s.partition[102]


def test_live_observer_cannot_edit_or_change_camera(session):
    import threading
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError
    from proofreading.server import make_server

    server = make_server(session, 0, read_only=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    before = session.state()
    try:
        assert b"live agent" in urlopen(base).read()
        headers = {
            "Authorization": "Bearer " + server.agent_token,
            "Content-Type": "application/json",
        }
        data = json.dumps(
            {"name": "set_view", "arguments": {"center_nm": [0, 0, 0]}}
        ).encode()
        with pytest.raises(HTTPError) as e:
            urlopen(Request(base + "/api/tool", data, headers))
        assert e.value.code == 403
        assert json.load(urlopen(Request(base + "/api/events", headers=headers))) == []
        assert before == session.state()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("seg_dtype,csr_dtype", [(np.uint64, np.int64), (np.int64, np.uint64)])
def test_import_large_mixed_integer_ids_remains_exact(tmp_path, seg_dtype, csr_dtype):
    from proofreading.import_world import import_world

    world = tmp_path / "world"
    world.mkdir()
    csr = world / "root_atoms_csr"
    csr.mkdir()
    # Adjacent large IDs collapse to identical float64 values. Enough distinct
    # IDs force NumPy's sort-based membership path instead of its small-set path.
    wanted = np.arange(1000, dtype=np.int64) * 1_000_000 + 92338636640651475
    distractors = wanted + 1
    np.save(world / "seg.npy", wanted.astype(seg_dtype).reshape(10, 10, 10))
    np.save(world / "em.npy", np.full((10, 10, 10), 120, np.uint8))
    root = 864691136698958190
    np.save(csr / "root_ids.npy", np.array([root, root + 1], csr_dtype))
    np.save(csr / "initial_roots.npy", np.array([root, root + 1], seg_dtype))
    np.save(csr / "indptr.npy", np.array([0, 1000, 2000]))
    np.save(csr / "atoms.npy", np.concatenate([wanted, distractors]).astype(csr_dtype))
    (world / "manifest.json").write_text(json.dumps({
        "resolution_xyz_nm": [32, 32, 40], "bounds_min_xyz": [0, 0, 0]
    }))
    path = import_world(world, tmp_path / "crop", [0, 0, 0, 10, 10, 10])
    dataset = Dataset.load(path)
    assert dataset.initial == {int(atom): root for atom in wanted}
    np.testing.assert_array_equal(dataset.atoms.data.ravel(), wanted)

    # A genuinely absent historical atom must still fail, before writing output.
    incomplete = np.concatenate([wanted[1:], distractors, [distractors[-1] + 1]])
    np.save(csr / "atoms.npy", incomplete.astype(csr_dtype))
    with pytest.raises(ValueError, match="1 missing atoms"):
        import_world(world, tmp_path / "incomplete", [0, 0, 0, 10, 10, 10])
    assert not (tmp_path / "incomplete").exists()


def test_persistent_3d_camera_is_reversible_on_invalid_input(session):
    call(session, 'select', segments=['12'])
    before=dict(session.partition)
    result=call(session,'set_scene',target_nm=[120,230,340],azimuth=65,elevation=40,view_height_nm=100)
    assert (session.output/result['image']).exists()
    assert result['state']['scene']['target_nm']==[120,230,340]
    old=dict(session.scene)
    assert not session.execute('set_scene',{'view_height_nm':-1})['ok']
    assert not session.execute('set_scene',{'elevation':90})['ok']
    assert session.scene==old and session.partition==before and session.revision==0


def test_em_observation_refreshes_scene(session):
    import copy
    camera = copy.deepcopy(session.scene)
    first = call(session, "look")
    view = copy.deepcopy(session.view.metadata())
    center = list(session.view.center_nm)
    center[2] += session.view.pixel_size_nm * 4
    second = call(session, "set_view", center_nm=center)
    companion = second["scene_observation"]
    assert companion["em_transform"] == second["transform"]
    assert companion["scene"] == camera
    assert companion["em_transform"] != view
    assert (session.output / first["scene_observation"]["image"]).read_bytes() != (session.output / companion["image"]).read_bytes()
    before = copy.deepcopy(session.partition)
    preview = call(session, "preview_merge", segments=["12", "13"])
    assert {str(session.next_id)} <= set(preview["scene_observation"]["selected"])
    assert session.partition == before


def test_incomplete_provider_response_not_reported_as_voluntary_finish(session):
    from proofreading.harness import run
    def transport(*args):
        return {'status':'incomplete','incomplete_details':{'reason':'max_output_tokens'},'output':[]}
    result=run(session,'test',api_key='test',transport=transport,max_actions=1)
    assert result['status']=='provider_incomplete'
    assert result['actions']==0


def test_anthropic_effort_and_truncated_output_do_not_execute_edits(session):
    from proofreading.harness import run
    def transport(endpoint,payload,headers):
        assert payload['output_config']=={'effort':'max'}
        assert payload['max_tokens']==65536
        return {'stop_reason':'max_tokens','content':[{'type':'tool_use','name':'finish','id':'partial','input':{'summary':'incomplete'}}]}
    report=run(session,'claude-opus-5-5',provider='anthropic',api_key='test',
               model_options={'output_config':{'effort':'max'}},max_output_tokens=65536,transport=transport)
    assert report['status']=='provider_incomplete'
    assert report['actions']==0
    assert not session.finished


def test_harness_explicit_system_profile_is_sent_and_saved(session):
    from proofreading.harness import run
    def transport(endpoint,payload,headers):
        assert payload['system']=='Frozen baseline\n\nTask objective:\nReview the starting process'
        return {'content':[{'type':'tool_use','id':'done','name':'finish','input':{'summary':'test'}}]}
    result=run(session,'test',provider='anthropic',api_key='test',transport=transport,
               system_prompt='Frozen baseline',objective='Review the starting process')
    assert result['status']=='finished'
    assert (session.output/'system-prompt.txt').read_text().startswith('Frozen baseline\n')


def test_seeded_split_preserves_disconnected_same_side(dataset, tmp_path):
    d = Dataset(dataset.atoms, dataset.em, {101: 1, 102: 1, 103: 1}, [(101, 102, 1.0)])
    s = Session(d, tmp_path / 'preserved')
    preview = call(s, 'preview_split', segment='1',
                   source_nm=[[110, 215, 320], [110, 245, 320]], sink_nm=[[124, 215, 320]])
    assert sorted(preview['child_atom_counts']) == [1, 2]
    assert preview['source_sink_component_counts'] == [2, 1]
    assert preview['unseeded_component_atom_counts'] == []
    call(s, 'commit', preview_id=preview['preview_id'])
    assert s.partition[101] == s.partition[103] != s.partition[102]
    from proofreading.replay import replay
    assert replay(d, s.output, tmp_path / 'preserved-replay')['partition_equal']


def test_seeded_split_reports_unsupported_islands_and_replays_legacy(dataset, tmp_path):
    d = Dataset(dataset.atoms, dataset.em, {101: 1, 102: 1, 103: 1}, [(101, 102, 1.0)])
    for policy, count in [('preserve_sides', 2), ('connected_components', 3)]:
        s = Session(d, tmp_path / policy, seeded_split_policy=policy)
        preview = call(s, 'preview_split', segment='1', source_nm=[[110, 215, 320]], sink_nm=[[124, 215, 320]])
        assert len(preview['child_atom_counts']) == count
        assert preview['unseeded_component_atom_counts'] == [1]
        call(s, 'commit', preview_id=preview['preview_id'])
        assert set(s.partition) == {101, 102, 103}
        if policy == 'connected_components':
            meta_path = s.output / 'session.json'
            meta = json.loads(meta_path.read_text()); del meta['seeded_split_policy']
            meta_path.write_text(json.dumps(meta))
            from proofreading.replay import replay
            assert replay(d, s.output, tmp_path / 'legacy-replay')['partition_equal']


def test_anthropic_cache_preserves_multiturn_vision_and_records_policy(session):
    from proofreading.harness import run
    import json
    seen=[]
    options={'cache_control':{'type':'ephemeral','ttl':'5m'}}
    def transport(endpoint,payload,headers):
        seen.append(json.loads(json.dumps(payload)))
        if len(seen)==1:
            return {'content':[{'type':'tool_use','id':'look1','name':'look','input':{}}]}
        return {'content':[{'type':'tool_use','id':'finish1','name':'finish','input':{'summary':'done'}}]}
    result=run(session,'claude-opus-5-5',provider='anthropic',api_key='test',transport=transport,model_options=options)
    assert len(seen)==2
    assert seen[1]['messages'][0]==seen[0]['messages'][0]
    assert any(b['type']=='image' for b in seen[0]['messages'][0]['content'])
    assert seen[1]['messages'][-1]['content'][0]['tool_use_id']=='look1'
    assert all(p['cache_control']==options['cache_control'] for p in seen)
    assert result['model_options']==options
    assert result['status']=='finished'


def test_openai_rejects_anthropic_cache_policy_before_inference(session):
    from proofreading.harness import run
    with pytest.raises(ValueError,match='Anthropic'):
        run(session,'gpt-6-astra',api_key='test',model_options={'cache_control':{'type':'ephemeral'}})


@pytest.mark.parametrize('policy',[{}, {'cache_control':None}])
def test_anthropic_default_cache_and_opt_out(session, policy):
    from proofreading.harness import run
    def transport(endpoint,payload,headers):
        assert ('cache_control' in payload)==('cache_control' not in policy)
        if 'cache_control' in payload:assert payload['cache_control']=={'type':'ephemeral','ttl':'5m'}
        return {'content':[{'type':'tool_use','id':'f','name':'finish','input':{'summary':'done'}}],
                'usage':{'input_tokens':10,'cache_read_input_tokens':100,'cache_creation_input_tokens':20,'output_tokens':5}}
    result=run(session,'claude-opus-5-5',provider='anthropic',api_key='test',transport=transport,model_options=policy)
    assert result['usage_summary']['input_tokens']==130
    assert result['usage_summary']['cache_read_tokens']==100
    assert policy==({} if not policy else {'cache_control':None})


def test_cache_accounting_does_not_double_count_writes_or_reasoning():
    from proofreading.usage import summarize_usage
    u=summarize_usage('responses',[{'input_tokens':100,'input_tokens_details':{'cached_tokens':70,'cache_write_tokens':20},'output_tokens':15,'output_tokens_details':{'reasoning_tokens':10}}])
    assert u['uncached_input_tokens']==10
    assert u['output_tokens']==15
    assert u['cache_read_fraction']==.7


def test_rounded_camera_directions_preserve_pixel_geometry(session):
    before=session.state()
    result=session.execute('set_view',{'right':[0,1,0],'down':[-.725,0,.689],'width':16,'height':16})
    assert result['ok'],result
    transform=result['result']['transform']
    r=np.asarray(transform['right']);d=np.asarray(transform['down'])
    assert np.isclose(np.linalg.norm(r),1) and np.isclose(np.linalg.norm(d),1)
    assert abs(r@d)<1e-12
    np.testing.assert_allclose(np.cross(r,d),[.689,0,.725]/np.linalg.norm([.689,0,.725]))
    original=session.view.metadata()
    session.view.validate();assert session.view.metadata()==original
    assert session.execute('undo_action',{})['ok'] and session.state()==before
