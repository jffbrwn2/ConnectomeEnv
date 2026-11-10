"""Provider-neutral JSON schemas. Integer IDs travel as decimal strings."""

S = {"type": "string"}
N = {"type": "number"}
I = {"type": "integer"}
B = {"type": "boolean"}
V = {"type": "array", "items": N, "minItems": 3, "maxItems": 3}
IDS = {"type": "array", "items": S}
POINTS = {"type": "array", "items": V}


def spec(description, properties=None, required=()):
    return {
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties or {},
            "required": list(required),
            "additionalProperties": False,
        },
    }


SCHEMAS = {
    "state": spec(
        "Current public state, camera and volume bounds; no hidden scoring information."
    ),
    "look": spec(
        "Render current EM view with segmentation overlay and a synchronized 3D scene showing its slice plane. Returns both images, exact pixel-to-world transform and view ID."
    ),
    "set_view": spec(
        "Returns synchronized EM and 3D images. Navigate anywhere in the volume. Set approximately perpendicular right/down directions to rotate or make oblique slices; rounded directions are normalized and the exact basis is returned. Pixel centers are zero-based; contrast is [low,high]. Slab clicks always select on the center plane.",
        {
            "center_nm": V,
            "right": V,
            "down": V,
            "pixel_size_nm": N,
            "width": I,
            "height": I,
            "contrast": {"type": "array", "items": N, "minItems": 2, "maxItems": 2},
            "gamma": N,
            "overlay": N,
            "slab_nm": N,
            "projection": {"type": "string", "enum": ["slice", "mean", "max", "min"]},
            "isolate": B,
        },
    ),
    "set_em_resolution": spec(
        "Choose matching EM and segmentation source resolution independently of zoom. Pass null to restore the default (automatic detail while zooming in streamed worlds; original grid in fixed worlds). See state.em_sampling.available_xy_nm; smaller xy_nm gives finer detail (e.g. 32 → 16 → 8), larger restores coarse detail. Z remains native. Returns fresh paired EM/3D images. Undoable with undo_action. Both layers change together; unmatched resolutions fail explicitly. This fetches source data, not artificial upscaling.",
        {"xy_nm": {"type": ["number", "null"], "exclusiveMinimum": 0}}, ["xy_nm"],
    ),
    "set_scene": spec(
        "Rotate, pan or zoom the persistent 3D selected-segment view within the fixed volume extent. target_nm is the camera look-at point; azimuth/elevation are degrees; view_height_nm is the vertical field of view (smaller zooms in). Returns a 3D image. The EM camera is independent.",
        {"target_nm": V, "azimuth": N, "elevation": {"type":"number","minimum":-89.9,"maximum":89.9}, "view_height_nm": {"type":"number","exclusiveMinimum":0}},
    ),
    "click": spec(
        "Pick/select/deselect/hide/show a segment at an EM or 3D image pixel using that image's view_id. In 3D, picks the frontmost visible surface and returns a verified interior voxel coordinate. Background or unavailable historical ownership returns hit=false without changing selection; stale images are rejected. Selection-changing clicks return refreshed EM and 3D observations; pick only identifies without changing selection. Use the returned world coordinate as a split seed.",
        {
            "view_id": S,
            "x": N,
            "y": N,
            "mode": {
                "type": "string",
                "enum": ["pick", "select", "replace", "deselect", "hide", "show"],
            },
        },
        ["view_id", "x", "y"],
    ),
    "select": spec(
        "Select, replace selection, deselect, hide or show known segments. Returns refreshed EM and 3D observations immediately. This does not edit segmentation.",
        {
            "segments": IDS,
            "mode": {
                "type": "string",
                "enum": ["select", "replace", "deselect", "hide", "show"],
            },
        },
        ["segments"],
    ),
    "list_segments": spec(
        "List only colored segments in the current EM view, paginated by visible pixel count. Includes stable colors shared with 3D. No graph or whole-volume scan.",
        {"offset": I, "limit": I},
    ),
    "annotate": spec(
        "Save an annotation at a physical XYZ point.",
        {"xyz_nm": V, "text": S},
        ["xyz_nm", "text"],
    ),
    "preview_merge": spec(
        "Preview the union of any chosen segments. Requires explicit commit; does not decide correctness.",
        {"segments": IDS},
        ["segments"],
    ),
    "preview_split": spec(
        "Preview a split using lists of physical source/sink seeds, OR a complete explicit partition of parent atoms. The default algorithm=connected_min_cut keeps same-side marks connected: it connects same-side marks along affinity paths and protects those paths while cutting. Place multiple marks along each intended process, using clicks from several angles if needed. Conflicting paths may require revised marks. algorithm=min_cut constrains only the seed atoms and can sever paths between same-side marks. Inspect both preview pieces before committing; neither algorithm proves anatomy. Seed coordinates use the current EM/segmentation resolution. Seeded cuts preserve two source/sink sides even if pieces are disconnected inside the crop. Unseeded disconnected pieces have uncertain side assignment: inspect or supply more seeds. Explicit groups preserve every supplied child. Cannot split inside an immutable atom.",
        {
            "segment": S,
            "source_nm": POINTS,
            "sink_nm": POINTS,
            "groups": {"type": "array", "items": IDS},
            "algorithm": {"type": "string", "enum": ["connected_min_cut", "min_cut"]},
        },
        ["segment"],
    ),
    "commit": spec(
        "Apply a previously previewed edit at its original revision. Stale previews are rejected.",
        {"preview_id": S},
        ["preview_id"],
    ),
    "undo_edit": spec("Undo the most recent committed split/merge and restore its pre-edit selection. Keep the current EM and 3D cameras. Returns fresh paired observations."),
    "undo_action": spec("Undo the latest successful state-changing action: camera/view, selection, annotation, preview, commit, or edit undo/redo. Skip read-only inspections and failed requests. Repeated calls walk backward. Returns the undone action and fresh paired observations."),
    "redo_edit": spec("Redo the most recently undone segmentation edit; keep current cameras."),
    "mesh": spec(
        "Inspect arbitrary selected segments as 3D surfaces. Control camera with azimuth and elevation in degrees. stride=1 is full resolution.",
        {"segments": IDS, "elevation": N, "azimuth": N, "stride": I},
    ),
    "finish": spec(
        "End the episode and give your final assessment. Does not expose ground truth.",
        {"summary": S},
        ["summary"],
    ),
}


SCHEMAS["look"]["parameters"] = {
    **SCHEMAS["set_view"]["parameters"],
    "properties": {**SCHEMAS["set_view"]["parameters"]["properties"],
                   "offsets_nm": {"type":"array","items":N,"minItems":1,"maxItems":9}},
}
SCHEMAS["look"]["description"] += " Optional view parameters render a temporary view without changing the current EM camera or action history. Supply offsets_nm to inspect a stack of single slices (same behavior as look_stack)."

SCHEMAS["look_stack"] = spec(
    "Inspect consecutive EM slices in one call. offsets_nm (one to nine values) shifts each plane along cross(right,down) from the current or supplied center, e.g. [-80,-40,0,40,80] for a native 40 nm Z stack in XY. Returns full paired EM/3D observations and independent clickable view IDs. Uses the current source resolution; does not change camera, selection, or undo history. Always single slices, not slab projections.",
    {**{k:v for k,v in SCHEMAS["set_view"]["parameters"]["properties"].items() if k not in {"slab_nm", "projection"}},
     "offsets_nm": {"type": "array", "items": N, "minItems": 1, "maxItems": 9}},
    ["offsets_nm"],
)

LEGACY_SCHEMAS = {
    "undo": spec("Legacy alias for undo_edit; historical replay only."),
    "redo": spec("Legacy alias for redo_edit; historical replay only."),
    "inspect_segment": spec(
        "Inspect segment atom membership and all graph neighbors; atom list is paginated.",
        {"segment": S, "offset": I, "limit": I},
        ["segment"],
    ),
}


def schemas():
    return [{"name": name, **value} for name, value in SCHEMAS.items()]
