"""Versioned task objectives for reproducible agent runs."""

AXON_EXTENSION = """Goal: reconstruct an axon by correcting split errors.

Explore the EM and choose a segment that appears to be a single isolated axon fragment.
Prefer one with an endpoint inside the volume, where a continuation can be investigated.
Record why you chose it. The initial camera location is only a browsing reference;
choose your own starting axon within the declared volume.

Follow that axon to an endpoint using EM slices and the 3D view.
Search nearby segments for its continuation. Trace the membrane and process through
successive slices; proximity or matching shape alone is insufficient evidence.
Inspect each candidate before adding it. If it contains an apparent false merge,
resolve that first so only the correct continuation is added.
Preview and commit supported merges. Check the result in EM, then continue along
the newly extended axon, following one path to its end before returning to other endpoints.

Distinguish a segmentation break from a true ending, an ambiguous crossing, or a
process leaving the volume. Do not infer connections outside the available data.
Explain candidate acceptances and rejections. Use undo_edit to reverse an incorrect
edit and undo_action to reverse navigation or selection. Finish with the reconstructed
segment, accepted continuations, and unresolved endpoints."""

# Experimental practice variant; keep the existing overnight baseline unchanged.
LOCAL_PROOFREADING = """Inspect the supplied EM neighborhood and segmentation.
Correct a split or merge error only when the imaging supports it. Leaving the
segmentation unchanged is appropriate when the evidence supports no correction.
You may inspect any available segment and transform, and increase EM resolution.
Explain the evidence for each committed edit and any remaining uncertainty."""

LOCAL_PROOFREADING_COMPARE_CONTINUATIONS = LOCAL_PROOFREADING + """

If you suspect a split error, investigate the location of the suspected break
before making an edit. Look for and inspect all plausible continuations there;
do not merge the first nearby segment merely because it looks plausible.
Compare the candidates using the EM evidence for continuity with the starting
process, inspecting additional slices or views as needed. Choose the best-supported
continuation, rather than treating each candidate in isolation. Briefly state
which candidates you considered and why the chosen one is more convincing.
If none of the candidates is convincing enough, or you cannot distinguish the
best candidate with confidence, leave the segmentation unchanged at that location
and report the uncertainty. A split error is not guaranteed to be present."""


LOCAL_PROOFREADING_MUTUAL_CONTINUATIONS = LOCAL_PROOFREADING + """

If you suspect a split error, investigate all plausible continuations at that
junction before editing. Evaluate each candidate independently against the
starting fragment and the EM evidence. In the 3D segment view, display only the
starting fragment A and one candidate B at a time. Use select with mode="replace"
and segments=[A, B] to replace the displayed selection for each comparison; do
not accumulate candidates or show all alternatives together. Use the EM view to
find other candidates, then inspect each pair separately through successive
slices and useful orientations. Proximity or apparent 3D alignment is insufficient.

Require mutual best fit at the same proposed junction: B must be the most
convincing continuation of A, and A must be the most convincing continuation of B.
For every proposed pair, also investigate B's plausible alternative partners at
that junction, displaying B with one alternative at a time. A continuation at
B's other end is not a competing partner for this junction. Do not assume the
reverse match merely because you preferred B when starting from A.

After the independent pairwise inspections, compare their evidence and briefly
state the alternatives assessed in both directions. Merge only if A and B are
each other's best-supported continuation and there is convincing EM continuity.
If either direction is ambiguous, another partner is more convincing, or none
has sufficient evidence, leave the segmentation unchanged and report uncertainty.
A split error is not guaranteed to be present."""


# Preserve the exact mutual-fit objective used by completed experiments.
LOCAL_PROOFREADING_ARTIFACT_AWARE = LOCAL_PROOFREADING_MUTUAL_CONTINUATIONS.replace(
    "Require mutual best fit at the same proposed junction:",
    """Split errors can occur at breaks, missing sections, distortions, or other EM
artifacts. Inspect the available EM on both sides of a suspected break and use
its broader context together with the original 3D segment shapes. Consider
trajectory, caliber, branching, and internal features across neighboring sections
and orientations. An obscured junction need not have a perfectly visible,
uninterrupted membrane or cytoplasmic channel to support a correction. Distinguish
missing or corrupted evidence from a clearly observed separating membrane.

Neuronal processes belong to cells and ordinarily connect through their arbor to
a cell body containing a nucleus. An isolated segmentation fragment may therefore
have a missing connection somewhere along it. This motivates looking for a
continuation; it does not establish that the split is at the current location.
The connection can lie outside the visible neighborhood or volume, and a process
can also have a genuine terminal endpoint. Do not merge a process directly into
a nucleus label merely because a cell-body connection is expected.

If the local evidence is insufficient, zoom out and inspect the broader extent
of the process. Rotate and pan the 3D view, follow branches toward endpoints or
the cell body when useful, and move the EM view to inspect relevant locations.
Request or navigate to additional available context rather than treating the
initial framing as the search boundary. Return to a suspected junction and zoom
in for detailed EM checks. Continue comparing candidates one pair at a time;
a wider field of view does not mean displaying all candidate segments together.
If the process extends beyond the loaded geometry, seek additional coverage
before interpreting that edge as a biological endpoint. Distinguish the visible
extent from the full process, and report any unavailable or uninspected extent.

Require mutual best fit at the same proposed junction:""",
).replace(
    """Merge only if A and B are
each other's best-supported continuation and there is convincing EM continuity.
If either direction is ambiguous, another partner is more convincing, or none
has sufficient evidence, leave the segmentation unchanged and report uncertainty.""",
    """Commit a merge when A and B are each other's clearly best-supported
continuation, substantially more convincing than the alternatives, and the
available EM context and original 3D morphology jointly support the connection.
This can justify a merge across an EM artifact even when direct continuity at
the damaged section cannot be verified. State what is obscured and which
contextual observations support the inference. A merely least-bad candidate,
proximity alone, or a connected merge preview is not sufficient. A clearly
observed separating membrane is evidence against merging. If the match remains
ambiguous in either direction or lacks adequate contextual support, leave it
unchanged and report uncertainty.""",
)


LOCAL_PROOFREADING_JUNCTION_EVIDENCE = LOCAL_PROOFREADING + """

Before merging, identify the proposed junction in physical coordinates. Follow
each candidate's membrane-bounded interior through successive EM slices into that
junction, and inspect a second plane to distinguish continuity from a crossing
or mere contact. State which observed sections support a shared interior. If a
separating membrane persists, keep the processes separate. If the data do not
resolve the junction, record the ambiguity and leave it unchanged.

A connected 3D shape after preview is not evidence of biological correctness: a
merge preview combines the requested segments by construction. Use it to check
which material the edit will affect. A candidate may carry a false merge elsewhere;
inspect its branches and isolate the supported continuation before adding it.
For a split, similarly use the separating membrane as evidence and verify that
all resulting pieces are retained. Check committed edits again in EM."""
