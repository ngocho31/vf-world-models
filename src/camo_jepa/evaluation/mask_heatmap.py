"""Put masked token indices back on the source frame (WM-12 heatmaps).

Token ``i`` of the flattened window belongs to transition ``t = i // 512``
(clip ``[t, t+1]``), patch ``p = i % 512``, row ``p // 32``, column ``p % 32``:
a 16x16 cell of the 512x256 model input. The loader resizes the whole frame
without cropping (``data/camo.py:153-160``), so a cell covers
``16 * W0/512`` x ``16 * H0/256`` pixels of the original frame, i.e. 60 x 96 px
on a 1920x1536 frame (x3.75 horizontally, x6 vertically).
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

from .region_metrics import PATCH_GRID

INPUT_SIZE = (512, 256)  # (width, height) of the model input
PATCH = 16


def token_to_cell(indices, grid: tuple[int, int] = PATCH_GRID):
    """Token indices -> ``(transition, row, col)`` arrays."""
    rows, cols = grid
    indices = np.asarray(indices, dtype=np.int64)
    patch = indices % (rows * cols)
    return indices // (rows * cols), patch // cols, patch % cols


def cell_to_box(row, col, source_size: tuple[int, int], input_size: tuple[int, int] = INPUT_SIZE, patch: int = PATCH):
    """``(x0, y0, x1, y1)`` of patch cells on the source frame; sizes are ``(width, height)``."""
    scale_x = source_size[0] / input_size[0]
    scale_y = source_size[1] / input_size[1]
    row, col = np.asarray(row, dtype=np.float64), np.asarray(col, dtype=np.float64)
    return col * patch * scale_x, row * patch * scale_y, (col + 1) * patch * scale_x, (row + 1) * patch * scale_y


def _cell_patches(rows, cols, source_size, scale, **style):
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    x0, y0, x1, y1 = cell_to_box(rows, cols, source_size)
    boxes = [Rectangle((a * scale, b * scale), (c - a) * scale, (d - b) * scale) for a, b, c, d in zip(x0, y0, x1, y1)]
    return PatchCollection(boxes, **style)


def _grid_outline(ax, cells: np.ndarray, color: str) -> None:
    from matplotlib.patches import Rectangle

    for row, col in zip(*np.nonzero(cells)):
        ax.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False, edgecolor=color, linewidth=0.7))


def _save_small_png(fig, out_path: Path, max_bytes: int) -> int:
    """Save as a palette PNG, dropping palette size until it fits ``max_bytes``."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100)
    image = Image.open(buffer).convert("RGB")
    for colors in (256, 128, 64):
        encoded = io.BytesIO()
        image.quantize(colors=colors, method=Image.Quantize.MEDIANCUT).save(encoded, format="PNG", optimize=True)
        if encoded.tell() <= max_bytes:
            out_path.write_bytes(encoded.getvalue())
            return encoded.tell()
    raise ValueError(f"heatmap {out_path.name} stays above {max_bytes} bytes")


def render_mask_heatmap(
    frame_path: str | Path, out_path: str | Path, *, mask_indices, cosines, dynamic, magnitude,
    transition: int, title: str, vmin: float, vmax: float, display_width: int = 960, max_bytes: int = 500_000,
) -> int:
    """Draw one transition of one window: masked cells coloured by cosine on the frame,
    dynamic cells outlined, plus the 16x32 cosine grid and the 16x32 |flow| grid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, cols = PATCH_GRID
    t_idx, r_idx, c_idx = token_to_cell(mask_indices)
    keep = t_idx == transition
    r_idx, c_idx, cos = r_idx[keep], c_idx[keep], np.asarray(cosines, dtype=np.float64)[keep]
    dyn_grid = np.asarray(dynamic, dtype=bool).reshape(-1, rows, cols)[transition]
    flow_grid = np.asarray(magnitude, dtype=np.float64).reshape(-1, rows, cols)[transition]
    cos_grid = np.full((rows, cols), np.nan)
    cos_grid[r_idx, c_idx] = cos

    frame = Image.open(frame_path).convert("RGB")
    source_size = frame.size
    scale = display_width / source_size[0]
    frame = frame.resize((display_width, round(source_size[1] * scale)), Image.Resampling.LANCZOS)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d9d9d9")
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(12.6, 6.4))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.12, 1.0), left=0.01, right=0.96, top=0.87, bottom=0.04, wspace=0.06, hspace=0.28)
    ax_frame = fig.add_subplot(grid[:, 0])
    ax_frame.imshow(frame)
    ax_frame.add_collection(_cell_patches(r_idx, c_idx, source_size, scale, facecolors=cmap(norm(cos)), alpha=0.55, edgecolors="none"))
    dyn_r, dyn_c = np.nonzero(dyn_grid)
    ax_frame.add_collection(_cell_patches(dyn_r, dyn_c, source_size, scale, facecolors="none", edgecolors="white", linewidths=0.9))
    ax_frame.set_title("masked cells coloured by cosine; white outline = dynamic (top 20% |flow|)", fontsize=9)
    ax_frame.axis("off")

    ax_cos = fig.add_subplot(grid[0, 1])
    shown = ax_cos.imshow(np.ma.masked_invalid(cos_grid), cmap=cmap, norm=norm, interpolation="nearest")
    _grid_outline(ax_cos, dyn_grid, "#e8175d")
    ax_cos.set_title("cosine (layer-normed) per masked token; grey = context, pink = dynamic", fontsize=9)
    fig.colorbar(shown, ax=ax_cos, fraction=0.025, pad=0.01)
    ax_flow = fig.add_subplot(grid[1, 1])
    shown = ax_flow.imshow(flow_grid, cmap="magma", interpolation="nearest")
    _grid_outline(ax_flow, dyn_grid, "#39d0ff")
    ax_flow.set_title("patch |flow| in input pixels (512x256); cyan = dynamic", fontsize=9)
    fig.colorbar(shown, ax=ax_flow, fraction=0.025, pad=0.01)
    for ax in (ax_cos, ax_flow):
        ax.set_xticks([])
        ax.set_yticks([])
    in_dynamic = dyn_grid[r_idx, c_idx]
    means = [f"{label} {cos[sel].mean():.3f} (n={int(sel.sum())})" if sel.any() else f"{label} n/a"
             for label, sel in (("dynamic", in_dynamic), ("static", ~in_dynamic))]
    fig.suptitle(f"{title}\nmean cosine on masked cells of this transition: " + ", ".join(means), fontsize=11)
    try:
        return _save_small_png(fig, Path(out_path), max_bytes)
    finally:
        plt.close(fig)


def render_heatmaps(captured: dict, frame_for, out_dir: Path, transition: int) -> list[dict]:
    """Render every captured window x mask with one shared colour range.

    ``captured`` maps window id to ``{"dynamic", "magnitude", "masks": {name: (mask, cos_ln)}}``;
    ``frame_for(window_id)`` gives the source frame of ``transition``. The colour range is the
    2nd-98th percentile of all rendered cosines, so figures compare with each other.
    """
    if not captured:
        return []
    values = np.concatenate([cos for tokens in captured.values() for _, cos in tokens["masks"].values()])
    vmin, vmax = (float(v) for v in np.percentile(values, [2, 98]))
    rendered = []
    for window_id, tokens in sorted(captured.items()):
        frame = Path(frame_for(window_id))
        for name, (mask, cosine) in tokens["masks"].items():
            path = Path(out_dir) / f"window{window_id:03d}_{name}.png"
            title = f"{name} masking | window {window_id} | frame {frame.stem} | transition {transition}->{transition + 1}"
            entry = {"file": path.name, "window": window_id, "mask": name, "frame": frame.stem, "cos_ln_colour_range": [vmin, vmax]}
            try:
                entry["bytes"] = render_mask_heatmap(frame, path, mask_indices=mask, cosines=cosine, dynamic=tokens["dynamic"],
                                                     magnitude=tokens["magnitude"], transition=transition, title=title,
                                                     vmin=vmin, vmax=vmax)
            except (OSError, ValueError) as error:  # a figure must not cost the scores already written
                entry["error"] = str(error)
            rendered.append(entry)
    return rendered
