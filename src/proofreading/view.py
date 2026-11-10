"""Orthographic views with explicit pixel-center-to-world transforms."""

from dataclasses import dataclass, asdict, field
import numpy as np
from PIL import Image
from .volume import vector


@dataclass
class View:
    center_nm: list
    right: list = field(default_factory=lambda: [1.0, 0.0, 0.0])
    down: list = field(default_factory=lambda: [0.0, 1.0, 0.0])
    pixel_size_nm: float = 40.0
    width: int = 512
    height: int = 512
    contrast: list = field(default_factory=lambda: [0.0, 255.0])
    gamma: float = 1.0
    overlay: float = 0.32
    slab_nm: float = 0.0
    projection: str = "slice"
    isolate: bool = False
    em_resolution_nm: float | None = None

    def validate(self):
        if self.em_resolution_nm is not None and (not np.isfinite(self.em_resolution_nm) or self.em_resolution_nm <= 0):
            raise ValueError("EM resolution must be positive")
        vector(self.center_nm, "center")
        r, d = vector(self.right, "right"), vector(self.down, "down")
        nr,nd=np.linalg.norm(r),np.linalg.norm(d)
        if not np.isfinite([nr,nd]).all() or min(nr,nd)<1e-12:
            raise ValueError("right and down must be nonzero finite direction vectors")
        unit_r,unit_d=r/nr,d/nd
        if abs(unit_r @ unit_d)>1e-2:
            raise ValueError("right and down must be approximately perpendicular world-space vectors")
        # Agents provide rounded directions. Normalize/orthogonalize once, then
        # use the exact reported basis for EM, 3D plane placement and clicking.
        if abs(nr-1)>1e-12 or abs(nd-1)>1e-12 or abs(r @ d)>1e-12:
            unit_d=unit_d-(unit_r @ unit_d)*unit_r
            self.right=unit_r.tolist()
            self.down=(unit_d/np.linalg.norm(unit_d)).tolist()
        if any(
            type(n) is not int or not 16 <= n <= 2048 for n in [self.width, self.height]
        ):
            raise ValueError("Image dimensions must be integers in [16,2048]")
        if not np.isfinite(self.pixel_size_nm) or self.pixel_size_nm <= 0:
            raise ValueError("pixel_size_nm must be positive")
        if not np.isfinite(self.gamma) or self.gamma <= 0 or not 0 <= self.overlay <= 1:
            raise ValueError("Invalid gamma or overlay")
        c = np.asarray(self.contrast)
        if c.shape != (2,) or not np.isfinite(c).all() or c[0] >= c[1]:
            raise ValueError("contrast requires finite low < high")
        if (
            not np.isfinite(self.slab_nm)
            or self.slab_nm < 0
            or self.projection not in ["slice", "mean", "max", "min"]
        ):
            raise ValueError("Invalid slab projection")
        return self

    def xyz(self, x, y):
        # x/y are zero-based pixel centers, including fractional clicks.
        return (
            np.asarray(self.center_nm)
            + (np.asarray(x)[..., None] - (self.width - 1) / 2)
            * self.pixel_size_nm
            * np.asarray(self.right)
            + (np.asarray(y)[..., None] - (self.height - 1) / 2)
            * self.pixel_size_nm
            * np.asarray(self.down)
        )

    def metadata(self):
        m = asdict(self)
        m["pixel_origin_nm"] = self.xyz(0, 0).tolist()
        m["pixel_right_nm"] = (np.asarray(self.right) * self.pixel_size_nm).tolist()
        m["pixel_down_nm"] = (np.asarray(self.down) * self.pixel_size_nm).tolist()
        m["click_semantics"] = (
            "Zero-based pixel centers; selection always samples the center plane, including slab views."
        )
        return m


def color(segment):
    # Stable across processes; segment IDs can exceed Javascript integer precision.
    import colorsys

    h = ((int(segment) * 11400714819323198485) & ((1 << 64) - 1)) / 2**64
    return np.array(colorsys.hsv_to_rgb(h, 0.62, 0.95)) * 255


def render(dataset, partition, selected, hidden, view, *, diagnostics=None):
    view.validate()
    yy, xx = np.mgrid[: view.height, : view.width]
    xyz = view.xyz(xx, yy)
    from .sampling import grids
    em, _ = grids(dataset, view)
    normal = np.cross(view.right, view.down)
    sampled=None
    if (view.projection == "slice" or view.slab_nm == 0) and hasattr(dataset,'sample_plane'):
        intensity,valid,atoms,inside=dataset.sample_plane(view,xyz)
        sampled=(atoms,inside & valid)
    elif view.projection == "slice" or view.slab_nm == 0:
        intensity, valid = em.linear(xyz)
    else:
        n = int(np.ceil(view.slab_nm / min(em.resolution))) + 1
        if n > 256:
            raise ValueError("Slab exceeds 256 samples; request a thinner slab")
        intensity = None
        total = np.zeros((view.height, view.width))
        valid = np.zeros((view.height, view.width), bool)
        for offset in np.linspace(-view.slab_nm / 2, view.slab_nm / 2, n):
            v, ok = em.linear(xyz + offset * normal)
            if intensity is None:
                intensity = (
                    np.zeros_like(v)
                    if view.projection == "mean"
                    else np.full_like(
                        v, np.inf if view.projection == "min" else -np.inf
                    )
                )
            if view.projection == "mean":
                intensity += np.where(ok, v, 0)
            elif view.projection == "min":
                intensity = np.minimum(intensity, np.where(ok, v, np.inf))
            else:
                intensity = np.maximum(intensity, np.where(ok, v, -np.inf))
            total += ok
            valid |= ok
        if view.projection == "mean":
            intensity /= np.maximum(total, 1)
        intensity = np.where(valid, intensity, 0)
    low, high = view.contrast
    if diagnostics is not None:
        values=intensity[valid]
        diagnostics.update(valid_pixels=int(values.size),source="sampled EM before contrast, gamma, and segmentation overlay")
        if values.size:
            diagnostics.update(
                percentiles={str(p):float(v) for p,v in zip((1,5,50,95,99),np.percentile(values,[1,5,50,95,99]))},
                fraction_clipped_low=float(np.mean(values<=low)),
                fraction_clipped_high=float(np.mean(values>=high)),
            )
    gray = (
        np.clip((intensity - low) / (high - low), 0, 1) ** (1 / view.gamma) * 255
    ).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=-1).astype(float)
    if sampled is None:
        _,atom_grid=grids(dataset,view)
        atoms,inside=atom_grid.nearest(xyz)
        _,em_inside=em.indices(xyz)
        sampled=(atoms,inside & em_inside)
    labels, inside, visible, counts = visible_segments(dataset, partition, selected, hidden, view, sampled=sampled)
    segment_ids, inverse = np.unique(labels, return_inverse=True)
    active = np.isin(segment_ids, np.asarray(visible, dtype=np.uint64))
    palette = np.array([color(int(s)) for s in segment_ids])
    opacity = np.array([view.overlay
                        for s in segment_ids])
    alpha = (opacity * active)[inverse].reshape(labels.shape) * inside
    rgb = rgb * (1-alpha[...,None]) + palette[inverse].reshape(rgb.shape) * alpha[...,None]
    # Missing historical ownership is not background and not a guessed segment.
    # Preserve EM, mark unavailable labeling explicitly, and exclude it from IDs.
    unknown=(sampled[0]!=0)&(labels==0)&inside if sampled is not None else np.zeros(labels.shape,bool)
    if diagnostics is not None:
        diagnostics['segmentation_coverage']={'unavailable_pixels':int(unknown.sum()),
            'note':'Magenta hatching marks unavailable historical ownership when overlay is enabled; these pixels cannot be selected or edited. EM remains visible.'}
    if view.overlay>0 and unknown.any():
        yy,xx=np.indices(labels.shape)
        hatch=unknown&((xx+yy)%8<2)
        rgb[hatch]=.5*rgb[hatch]+.5*np.array([255,80,190])
    visible = list(map(str, visible))
    # Out-of-data pixels are conspicuous and unclickable.
    rgb[~valid] = [34, 39, 49]
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)), visible


def visible_segments(dataset, partition, selected, hidden, view, *, sampled=None):
    """Only colored foreground on the current EM center plane; no global graph scan."""
    view.validate()
    if sampled is None:
        yy,xx=np.mgrid[:view.height,:view.width]
        xyz=view.xyz(xx,yy)
        from .sampling import grids
        em_grid,atom_grid=grids(dataset,view)
        atoms,inside=atom_grid.nearest(xyz)
        _,em_inside=em_grid.indices(xyz)
        inside=inside & em_inside
    else:
        # render already intersected the two source-validity masks.
        atoms,inside=sampled
    atom_ids,inverse=np.unique(atoms,return_inverse=True)
    owners=np.asarray(partition.display_owners(atom_ids) if hasattr(partition,"display_owners") else [partition[int(a)] if a else 0 for a in atom_ids],dtype=np.uint64)
    labels=owners[inverse].reshape(atoms.shape)
    ids,counts=np.unique(labels[inside],return_counts=True)
    visible=[int(s) for s in ids if s and int(s) not in hidden and (not selected or int(s) in selected)]
    if view.overlay==0:
        visible=[]
    return labels,inside,visible,{int(s):int(n) for s,n in zip(ids,counts)}
