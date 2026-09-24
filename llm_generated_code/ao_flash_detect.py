"""Detect AO flashes in raw TIFF stacks using fixed cells from timepoint zero.

Run from the repository root:
    python draw_figures/ao_flash_detect.py

Extend a saved 50-frame run, reusing masks and candidate detections:
    python draw_figures/ao_flash_detect.py --frames 60 --extend-from outputs/ao_flash_detect --output-dir outputs/ao_flash_detect_60frames

Defaults: all 35 stacks in intensity25/35/45/55, C2 flashes, C2 segmentation,
timepoints 0..49. Output goes to outputs/ao_flash_detect (raw data untouched).
Frequency = new spatially linked events / segmented cells / timepoint.
t=0 has no preceding baseline and is recorded as unavailable, not zero.

Adapted from flash_detection_code/pipeline/left_panel_revision/detect_left.py
and detect_slow.py: Gaussian smoothing, one/two-frame and delayed backgrounds,
local-maximum motion suppression, core/grow support, spatial/temporal linking.
RGB specificity, movie borders, manual edits and display persistence are omitted.
Intensity scaling uses ONLY C2 at t=0 (p1/p99.5), stays fixed across time, and
does not clip high values. The same normalized thresholds apply to all lasers.
These are automatic brightening candidates; inspect saved QA images before
biological interpretation. Spatial sizes below are in downsampled pixels.
"""
import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
import tifffile

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path(r"D:/cxlu/smart/final_data/20250604 AO")
EVENT_COLUMNS = ["laser", "stack", "event_id", "cell_id", "onset_t",
                 "last_detection_t", "x_px", "y_px", "x1_px", "y1_px",
                 "x2_px", "y2_px", "peak_rise_scaled", "peak_rise_raw",
                 "candidate_count"]
CANDIDATE_COLUMNS = ["laser", "stack", "t", "cell_id", "event_id", "x", "y",
                     "x1", "y1", "x2", "y2", "core_area", "grow_area",
                     "peak_rise", "pass_name"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output-dir", type=Path, default=REPO / "outputs/ao_flash_detect")
    p.add_argument("--conditions", nargs="+", type=int, default=[25, 35, 45, 55])
    p.add_argument("--frames", type=int, default=50, help="Maximum timepoints, including t=0 (at least 2)")
    p.add_argument("--flash-channel", type=int, default=2, help="One-based green channel (default: 2)")
    p.add_argument("--seg-channel", type=int, default=2, help="One-based channel for t=0 segmentation")
    p.add_argument("--model", type=Path,
                   default=Path.home() / ".cellpose/models/model_0528_resize")
    p.add_argument("--extend-from", type=Path, help="Reuse masks and candidates from a shorter run; use a new output directory")
    p.add_argument("--plot-only", action="store_true", help="Redraw saved results without detection")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--downsample", type=int, default=4)
    p.add_argument("--min-cell-area", type=int, default=40000, help="Full-resolution pixels")
    p.add_argument("--exclude-border-cells", action="store_true")
    p.add_argument("--fast-rise", type=float, default=14.0)
    p.add_argument("--slow-rise", type=float, default=22.0)
    p.add_argument("--relative-rise", type=float, default=0.075)
    p.add_argument("--link-distance", type=float, default=22.0)
    p.add_argument("--max-gap", type=int, default=4, help="Maximum separation of linked candidate timepoints")
    p.add_argument("--no-slow", action="store_true")
    p.add_argument("--max-stacks", type=int, help="Optional smoke-test limit PER condition")
    args = p.parse_args(argv)
    if args.frames < 2:
        p.error("--frames must be at least 2")
    if min(args.flash_channel, args.seg_channel, args.downsample, args.min_cell_area, args.max_gap) < 1:
        p.error("Channel, downsample, cell area and max gap must be positive")
    if min(args.fast_rise, args.slow_rise, args.link_distance) <= 0 or args.relative_rise < 0:
        p.error("Invalid detection thresholds")
    if args.max_stacks is not None and args.max_stacks < 1:
        p.error("--max-stacks must be positive")
    if len(set(args.conditions)) != len(args.conditions):
        p.error("Duplicate laser conditions")
    return args


def load_stack(path, args):
    """Decode only the flash channel for the requested times and the segmentation channel at t=0.

    TIFF series axes determine page indexing, including CTYX and singleton Z.
    Unknown non-singleton axes are rejected instead of guessing channel order.
    """
    with tifffile.TiffFile(path) as tf:
        series = tf.series[0]
        axes, shape = series.axes, series.shape
        if not axes.endswith("YX") or len(axes) != len(shape):
            raise ValueError("Expected planar TIFF with YX last; got %s %s" % (axes, shape))
        dims = dict(zip(axes, shape))
        if "T" not in dims or "C" not in dims:
            raise ValueError("Explicit T and C axes required; got %s %s" % (axes, shape))
        if any(size != 1 for ax, size in zip(axes[:-2], shape[:-2]) if ax not in "TC"):
            raise ValueError("Non-singleton Z/position axes are not supported")
        if dims["C"] < max(args.seg_channel, args.flash_channel):
            raise ValueError("Requested channel does not exist")
        nt = min(args.frames, dims["T"])
        if nt < 2:
            raise ValueError("At least two timepoints are required")
        h, w = shape[-2:]
        small_size = (max(1, w // args.downsample), max(1, h // args.downsample))

        def page_at(t, c):
            indices = tuple(t if ax == "T" else c if ax == "C" else 0 for ax in axes[:-2])
            idx = int(np.ravel_multi_index(indices, shape[:-2]))
            page = series.pages[idx]
            if page is None:
                raise ValueError("Missing TIFF plane t=%s c=%s" % (t, c))
            frame = page.asarray()
            if frame.shape != (h, w):
                raise ValueError("Unexpected/missing image plane")
            return frame

        seg = page_at(0, args.seg_channel - 1)
        green, saturation = [], []
        for t in range(nt):
            raw = page_at(t, args.flash_channel - 1)
            saturation.append(float(np.mean(raw == np.iinfo(raw.dtype).max))
                              if np.issubdtype(raw.dtype, np.integer) else 0.0)
            green.append(cv2.resize(raw.astype(np.float32), small_size,
                                    interpolation=cv2.INTER_AREA))
    return np.stack(green), seg, dict(axes=axes, shape=list(shape),
                                     n_frames=nt, saturation_fraction=saturation)


def make_model(args):
    if not args.model.is_file():
        raise FileNotFoundError("Missing trained segmentation model: %s" % args.model)
    # Import only the model module: ferroptosis_rsl3_ao.py executes on import.
    sys.path.insert(0, str(REPO / "cellquant"))
    from cellquant import models, core
    return models.CellposeModel(gpu=(not args.cpu and core.use_gpu()),
                                pretrained_model=str(args.model))


def segment_first_frame(image, small_shape, model, args):
    """Match the existing AO model preprocessing; evaluate exactly once."""
    lo, hi = np.percentile(image, [2, 98])
    if hi <= lo:
        raise ValueError("Flat first segmentation frame")
    normalized = np.clip((image.astype(np.float32) - lo) * 255 / (hi - lo), 0, 255)
    small = cv2.resize(normalized.astype(np.uint8), small_shape[::-1],
                       interpolation=cv2.INTER_LINEAR)
    labels = model.eval([small], channels=[0, 0], diameter=model.diam_labels,
                        flow_threshold=0.4, cellprob_threshold=0)[0][0].astype(np.int32)
    full = cv2.resize(labels, image.shape[::-1], interpolation=cv2.INTER_NEAREST)
    ids, areas = np.unique(full, return_counts=True)
    keep = ids[(ids > 0) & (areas >= args.min_cell_area)]
    if args.exclude_border_cells:
        border = np.unique(np.concatenate([full[0], full[-1], full[:, 0], full[:, -1]]))
        keep = keep[~np.isin(keep, border)]
    full[~np.isin(full, keep)] = 0
    labels[~np.isin(labels, keep)] = 0
    if not len(keep):
        raise ValueError("No accepted cells at t=0; inspect segmentation settings")
    return full, labels, keep


def smooth_cells(scaled, labels, sigma):
    """Normalized convolution within each cell excludes background/neighbor light."""
    smoothed = np.zeros_like(scaled, dtype=np.float32)
    for cell, sl in enumerate(ndi.find_objects(labels), 1):
        if sl is None:
            continue
        mask = (labels[sl] == cell).astype(np.float32)
        weight = ndi.gaussian_filter(mask, sigma=sigma, mode="constant")
        numerator = ndi.gaussian_filter(scaled[:, sl[0], sl[1]] * mask,
                                        sigma=(0, sigma, sigma), mode="constant")
        cell_smooth = numerator / np.maximum(weight, 1e-12)
        view = smoothed[:, sl[0], sl[1]]
        view[:, mask > 0] = cell_smooth[:, mask > 0]
    return smoothed


def detect_candidates(raw, labels, args, start_t=1):
    """Fast and slow passes share support regions to avoid duplicate counts."""
    lo, hi = np.percentile(raw[0], [1, 99.5])
    if hi <= lo:
        raise ValueError("Flat first detection frame")
    scale = float((hi - lo) / 255.0)
    scaled = (raw - lo) / scale  # No per-frame rescaling or high-value clipping.
    fast = smooth_cells(scaled, labels, 1.2)
    slow = None if args.no_slow else smooth_cells(scaled, labels, 2)
    result = []
    for t in range(max(1, start_t), len(raw)):
        prev = fast[max(0, t - 2):t].min(axis=0)
        dg = fast[t] - ndi.maximum_filter(prev, size=3)
        fc = ((dg > args.fast_rise) &
              (dg > args.relative_rise * np.maximum(prev, 30)) & (fast[t] > 35))
        fg = (dg > args.fast_rise / 2) & (fast[t] > 28)
        sc, sg = np.zeros_like(fc), np.zeros_like(fc)
        ds = np.zeros_like(dg)
        if slow is not None and t >= 3:
            baseline = slow[max(0, t - 7):max(1, t - 3)].mean(axis=0)
            ds = slow[t] - ndi.maximum_filter(baseline, size=3)
            sc = (ds > args.slow_rise) & (slow[t] > 40)
            sg = (ds > args.slow_rise * 12 / 22) & (slow[t] > 30)
        # Label separately within each fixed cell, never a shared crop rectangle.
        for cell in np.unique(labels[labels > 0]):
            cell_mask = labels == cell
            support, _ = ndi.label((fg | sg) & cell_mask)
            for j, sl in enumerate(ndi.find_objects(support), 1):
                if sl is None:
                    continue
                region = support[sl] == j
                cf, cs = fc[sl] & region, sc[sl] & region
                af, ass = int(np.sum(fg[sl] & region)), int(np.sum(sg[sl] & region))
                fast_ok = (cf.sum() >= 12 and af >= 28 and
                           np.percentile(dg[sl][cf], 90) >= args.fast_rise * 22 / 14)
                slow_ok = (cs.sum() >= 30 and ass >= 50 and
                           np.percentile(ds[sl][cs], 90) >= args.slow_rise * 30 / 22)
                if not (fast_ok or slow_ok):
                    continue
                core = (cf if fast_ok else np.zeros_like(cf)) | (cs if slow_ok else False)
                rise = np.maximum(dg[sl] if fast_ok else 0, ds[sl] if slow_ok else 0)
                yy, xx = np.where(core)
                weights = rise[core]
                result.append(dict(t=t, cell_id=int(cell),
                                   x=float(np.average(xx, weights=weights) + sl[1].start),
                                   y=float(np.average(yy, weights=weights) + sl[0].start),
                                   x1=sl[1].start, y1=sl[0].start,
                                   x2=sl[1].stop, y2=sl[0].stop,
                                   core_area=int(core.sum()), grow_area=int(region.sum()),
                                   peak_rise=float(weights.max()),
                                   pass_name="both" if fast_ok and slow_ok else
                                   "fast" if fast_ok else "slow"))
    return result, scaled, dict(baseline_p1=float(lo), baseline_p995=float(hi),
                               raw_units_per_scaled_unit=scale)


def link_events(candidates, args):
    """One-to-one temporal matching per cell, using a fixed onset position.

    A continuing candidate is counted once at its first detection. A repeat at
    the same site after a gap > max_gap is a new event (as in the source code).
    """
    events = []
    for t in sorted(set(c["t"] for c in candidates)):
        current = [c for c in candidates if c["t"] == t]
        active = [e for e in events if 0 < t - e["last_detection_t"] <= args.max_gap]
        matched = {}
        if active and current:
            costs = np.full((len(active), len(current)), 1e9)
            for i, e in enumerate(active):
                for j, c in enumerate(current):
                    distance = np.hypot(e["x"] - c["x"], e["y"] - c["y"])
                    if e["cell_id"] == c["cell_id"] and distance <= args.link_distance:
                        costs[i, j] = distance
            rows, cols = linear_sum_assignment(costs)
            matched = {j: active[i] for i, j in zip(rows, cols) if costs[i, j] < 1e9}
        for j, c in enumerate(current):
            if j in matched:
                e = matched[j]
                e["last_detection_t"] = t
                e["peak_rise"] = max(e["peak_rise"], c["peak_rise"])
                e["candidate_count"] += 1
                for key in ("x1", "y1"):
                    e[key] = min(e[key], c[key])
                for key in ("x2", "y2"):
                    e[key] = max(e[key], c[key])
            else:
                e = {k: c[k] for k in ("cell_id", "x", "y", "x1", "y1", "x2", "y2", "peak_rise")}
                e.update(event_id=len(events) + 1, onset_t=t, last_detection_t=t,
                         candidate_count=1)
                events.append(e)
            c["event_id"] = e["event_id"]
    return events


def save_qa(out, seg, labels, scaled, events):
    lo, hi = np.percentile(seg, [2, 98])
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(seg, cmap="gray", vmin=lo, vmax=hi)
    full = cv2.resize(labels, seg.shape[::-1], interpolation=cv2.INTER_NEAREST)
    boundary = (ndi.maximum_filter(full, size=3) != ndi.minimum_filter(full, size=3))
    overlay = np.zeros(full.shape + (4,), dtype=np.float32)
    overlay[boundary] = [0, 1, 1, 0.85]
    ax.imshow(overlay)
    for cell in np.unique(full[full > 0]):
        y, x = ndi.center_of_mass(full == cell)
        ax.text(x, y, str(cell), color="yellow", ha="center", fontsize=9)
    ax.set_title("t=0 fixed cells (labels used for every timepoint)")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out / "cell_masks_t0.png", dpi=130)
    plt.close(fig)
    # Before/onset crops for EVERY event; fixed display scaling for each pair.
    for start in range(0, len(events), 12):
        batch = events[start:start + 12]
        fig, axes = plt.subplots(len(batch), 2, figsize=(7, 2.1 * len(batch)), squeeze=False)
        for row, e in enumerate(batch):
            x, y, t = round(e["x"]), round(e["y"]), e["onset_t"]
            x1, x2 = max(0, x - 40), min(scaled.shape[2], x + 41)
            y1, y2 = max(0, y - 30), min(scaled.shape[1], y + 31)
            pair = scaled[t - 1:t + 1, y1:y2, x1:x2]
            vmin, vmax = np.percentile(pair, [1, 99.5])
            for col, frame in enumerate((t - 1, t)):
                ax = axes[row, col]
                ax.imshow(scaled[frame, y1:y2, x1:x2], cmap="gray", vmin=vmin, vmax=vmax)
                ax.plot(e["x"] - x1, e["y"] - y1, "+", color="red")
                ax.set_title("event %s | cell %s | t=%s" % (e["event_id"], e["cell_id"], frame),
                             fontsize=9)
                ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out / ("events_review_%03d.png" % (start // 12 + 1)), dpi=110)
        plt.close(fig)
    fig, axes = plt.subplots(2, 5, figsize=(16, 5.5))
    for ax, t in zip(axes.flat, np.linspace(0, len(scaled) - 1, 10).astype(int)):
        ax.imshow(scaled[t], cmap="gray", vmin=0, vmax=255)
        for e in events:
            if e["onset_t"] == t:
                ax.add_patch(Rectangle((e["x1"], e["y1"]), e["x2"] - e["x1"],
                                       e["y2"] - e["y1"], fill=False, edgecolor="red"))
        ax.set_title("t=%s" % t)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out / "timepoint_overview.png", dpi=130)
    plt.close(fig)


def summarize(cell_counts, frames):
    stack = cell_counts.groupby(["laser", "stack", "t"], as_index=False).agg(
        new_events=("new_events", lambda x: x.sum(min_count=1)),
        n_cells=("cell_id", "size"),
        cells_with_new_event=("new_events", lambda x: np.nan if x.isna().all() else (x > 0).sum()))
    stack["frequency"] = stack.new_events / stack.n_cells
    rows = []
    for laser in sorted(stack.laser.unique()):
        for t in range(frames):
            part = stack[(stack.laser == laser) & (stack.t == t)]
            n_cells = int(part.n_cells.sum())
            evaluable = t > 0 and n_cells > 0
            count = int(part.new_events.sum()) if evaluable else np.nan
            rates = part.frequency.dropna()
            rows.append(dict(laser=laser, t=t, n_stacks=len(part), n_cells=n_cells,
                             new_events=count, frequency=count / n_cells if evaluable else np.nan,
                             mean_stack_frequency=rates.mean(),
                             sem_stack_frequency=rates.sem() if len(rates) > 1 else np.nan,
                             fraction_cells_with_new_event=part.cells_with_new_event.sum() / n_cells
                             if evaluable else np.nan))
    return stack, pd.DataFrame(rows)


def plot_statistics(stack, cumulative=False):
    """Cell-weighted mean with stack-cluster SEM and Student-t 95% CI.

    The mean matches pooled events / cells. Treat each acquisition stack as
    an independent cluster, allowing dependence among cells in the same stack:
    SEM^2 = n/(n-1) * sum((w_i * (rate_i - mean))^2), w_i=cells_i/sum(cells).
    For equal cell counts this reduces to the ordinary across-stack SEM.
    Cumulative uncertainty is computed AFTER accumulating each stack's rates;
    pointwise SEM values are never summed.
    """
    from scipy.stats import t as student_t

    data = stack.sort_values(["laser", "stack", "t"]).copy()
    if cumulative:
        data["value"] = data.groupby(["laser", "stack"]).frequency.cumsum()
    else:
        data["value"] = data.frequency
    rows = []
    for (laser, timepoint), group in data.groupby(["laser", "t"]):
        valid = group[np.isfinite(group.value) & (group.n_cells > 0)]
        n = len(valid)
        mean = sem = halfwidth = np.nan
        if n:
            weights = valid.n_cells.to_numpy(dtype=float)
            weights /= weights.sum()
            values = valid.value.to_numpy(dtype=float)
            mean = float(np.dot(weights, values))
            if n > 1:
                sem = float(np.sqrt(n / (n - 1) * np.sum((weights * (values - mean)) ** 2)))
                halfwidth = float(student_t.ppf(0.975, df=n - 1) * sem)
        rows.append(dict(laser=laser, t=timepoint, n_stacks=n, mean=mean, sem=sem,
                         ci95_lower=mean - halfwidth, ci95_upper=mean + halfwidth))
    return pd.DataFrame(rows)


def save_plots(summary, out, stack=None):
    """Original presentation for frequency and cumulative plots."""
    colors = {25: "#377eb8", 35: "#4daf4a", 45: "#ff8c00", 55: "#e41a1c"}
    config = dict(style="original", colors_hex=colors, legend=True, grid=True,
                  uncertainty_conditions=[], marker="o", markersize=2.5,
                  linewidth=1.6, figsize=[9, 5],
                  note="CI and SEM remain available in plot-data CSVs but are not displayed.")
    (out / "plot_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    for cumulative in (False, True):
        fig, ax = plt.subplots(figsize=(9, 5))
        for laser, group in summary.groupby("laser"):
            # Keep unavailable timepoints as NaN.
            values = group.frequency.cumsum() if cumulative else group.frequency
            ax.plot(group.t, values, color=colors.get(laser), label=str(laser),
                    lw=1.6, marker="o", markersize=2.5)
        ax.set(xlabel="Timepoint (0-based)",
               ylabel="Cumulative events per cell" if cumulative else
               "Flash event frequency (new events / cell / timepoint)",
               xlim=(0, int(summary.t.max())), ylim=(0, None))
        ax.legend(title="Laser intensity", frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        name = "flash_cumulative" if cumulative else "flash_frequency"
        for ext in ("png", "pdf"):
            fig.savefig(out / (name + "." + ext), dpi=300)
        plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    args.data_root = args.data_root.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir == args.data_root or args.data_root in args.output_dir.parents:
        raise ValueError("Use an output directory outside the raw data tree")
    if args.plot_only:
        summary = pd.read_csv(args.output_dir / "frequency_by_condition.csv")
        save_plots(summary, args.output_dir)
        print("Updated plots in %s" % args.output_dir)
        return
    if args.extend_from:
        args.extend_from = args.extend_from.resolve()
        if (args.output_dir == args.extend_from or
                args.extend_from in args.output_dir.parents or
                args.output_dir in args.extend_from.parents):
            raise ValueError("Extension output must be separate from the previous results")
        previous_config = json.loads((args.extend_from / "run_config.json").read_text())
        for key in ("flash_channel", "seg_channel", "downsample", "min_cell_area",
                    "exclude_border_cells", "fast_rise", "slow_rise", "relative_rise",
                    "link_distance", "max_gap", "no_slow"):
            if getattr(args, key) != previous_config[key]:
                raise ValueError("Extension must keep original setting: %s" % key)
        if args.frames <= previous_config["frames"]:
            raise ValueError("--frames must exceed the previous run's frame count")
    inputs = []
    for laser in args.conditions:
        folder = args.data_root / ("intensity%s" % laser)
        files = sorted(set(folder.rglob("*.ome.tif")) | set(folder.rglob("*.ome.tiff")))
        if not files:
            raise FileNotFoundError("No raw OME-TIFF stacks under %s" % folder)
        inputs.extend((laser, path) for path in files[:args.max_stacks])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(created_at=datetime.now().isoformat(), flash_channel_one_based=args.flash_channel,
                  frequency_definition="new linked events / t=0 cells / timepoint",
                  timepoint_zero="baseline only; onset frequency unavailable",
                  coordinate_convention="events: raw pixels; candidates: downsampled pixels",
                  condition_assignment="parent intensity folder, not filename",
                  intensity_normalization="flash channel t=0 p1/p99.5 fixed over time, unclipped",
                  source_scripts=["detect_left.py", "detect_slow.py"],
                  inputs=[str(p) for _, p in inputs],
                  versions={m.__name__: m.__version__ for m in (np, pd, cv2, tifffile, matplotlib)})
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    model = None if args.extend_from else make_model(args)
    all_counts, all_events, stack_info, failures = [], [], [], []
    for index, (laser, path) in enumerate(inputs, 1):
        stack_id = path.relative_to(args.data_root).as_posix()
        print("[%s/%s] %s" % (index, len(inputs), stack_id), flush=True)
        out = args.output_dir / path.relative_to(args.data_root).parent / path.name.replace(".tif", "")
        out.mkdir(parents=True, exist_ok=True)
        try:
            raw, seg, info = load_stack(path, args)
            old_candidates, previous = [], None
            start_t = 1
            if args.extend_from:
                prior_dir = args.extend_from / out.relative_to(args.output_dir)
                previous = json.loads((prior_dir / "stack_info.json").read_text())
                if previous["stack"] != stack_id or previous["laser"] != laser:
                    raise ValueError("Saved stack identity does not match")
                if (previous["source_bytes"] != path.stat().st_size or
                        previous["source_mtime_ns"] != path.stat().st_mtime_ns):
                    raise ValueError("Raw stack changed since the previous analysis")
                start_t = previous["n_frames"]
                if len(raw) < args.frames:
                    raise ValueError("Stack has fewer than the requested timepoints")
                if start_t >= len(raw):
                    raise ValueError("No new timepoints for this stack")
                full = tifffile.imread(prior_dir / "cell_masks_t0.tif").astype(np.int32)
                if full.shape != seg.shape:
                    raise ValueError("Saved mask shape does not match raw data")
                # Original dimensions are exact multiples of downsample, so this
                # recovers the original analysis mask without changing its labels.
                if (seg.shape[0] % raw.shape[1] or seg.shape[1] % raw.shape[2]):
                    raise ValueError("Cannot exactly recover downsampled mask")
                labels = cv2.resize(full, raw.shape[:0:-1], interpolation=cv2.INTER_NEAREST)
                cells = np.unique(full[full > 0])
                if len(cells) != previous["n_cells"]:
                    raise ValueError("Saved cell count does not match mask")
                old_table = pd.read_csv(prior_dir / "candidates.csv", float_precision="round_trip")
                fields = [key for key in CANDIDATE_COLUMNS if key not in ("laser", "stack")]
                old_candidates = old_table[fields].to_dict("records")
                if any(c["t"] >= start_t or c["t"] < 1 for c in old_candidates):
                    raise ValueError("Invalid timepoint in previous candidates")
            else:
                full, labels, cells = segment_first_frame(seg, raw.shape[1:], model, args)
            tifffile.imwrite(out / "cell_masks_t0.tif", full.astype(np.uint32))
            new_candidates, scaled, normalization = detect_candidates(raw, labels, args, start_t=start_t)
            if previous:
                for key, value in normalization.items():
                    np.testing.assert_allclose(value, previous[key], rtol=1e-12)
            candidates = old_candidates + new_candidates
            saved_event_ids = [c["event_id"] for c in old_candidates]
            events = link_events(candidates, args)
            if previous:
                if [c["event_id"] for c in old_candidates] != saved_event_ids:
                    raise ValueError("Historical event IDs changed during extension")
                info.update(extended_from=str(prior_dir), reused_frames=start_t,
                            newly_detected_frames=len(raw) - start_t,
                            new_candidate_count=len(new_candidates))
            sx, sy = seg.shape[1] / labels.shape[1], seg.shape[0] / labels.shape[0]
            exported = []
            for e in events:
                row = {k: e[k] for k in ("event_id", "cell_id", "onset_t", "last_detection_t",
                                         "candidate_count")}
                row.update(laser=laser, stack=stack_id, peak_rise_scaled=e["peak_rise"],
                           peak_rise_raw=e["peak_rise"] * normalization["raw_units_per_scaled_unit"])
                for k in ("x", "y", "x1", "y1", "x2", "y2"):
                    row[k + "_px"] = e[k] * (sx if k.startswith("x") else sy)
                exported.append(row)
            pd.DataFrame(exported, columns=EVENT_COLUMNS).to_csv(out / "events.csv", index=False)
            pd.DataFrame([dict(laser=laser, stack=stack_id, **c) for c in candidates],
                         columns=CANDIDATE_COLUMNS).to_csv(out / "candidates.csv", index=False)
            rows = []
            for cell in cells:
                for t in range(len(raw)):
                    rows.append(dict(laser=laser, stack=stack_id, cell_id=int(cell), t=t,
                                     new_events=np.nan if t == 0 else
                                     sum(e["cell_id"] == cell and e["onset_t"] == t for e in events)))
            counts = pd.DataFrame(rows)
            if previous:
                old_counts = pd.read_csv(prior_dir / "cell_timepoints.csv")
                keys = ["cell_id", "t"]
                observed = counts[counts.t < start_t].sort_values(keys).reset_index(drop=True)
                expected = old_counts.sort_values(keys).reset_index(drop=True)
                pd.testing.assert_frame_equal(observed, expected, check_dtype=False,
                                              check_exact=False, rtol=1e-12, atol=1e-12)
            counts.to_csv(out / "cell_timepoints.csv", index=False)
            info.update(normalization, laser=laser, stack=stack_id, source=str(path),
                        source_bytes=path.stat().st_size, source_mtime_ns=path.stat().st_mtime_ns,
                        n_cells=len(cells), n_events=len(events), scale_x=sx, scale_y=sy)
            (out / "stack_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
            save_qa(out, seg, labels, scaled, events)
            all_counts.append(counts)
            all_events.extend(exported)
            stack_info.append({k: info[k] for k in ("laser", "stack", "n_frames", "n_cells", "n_events")})
            print("  cells=%s; events=%s; frames=%s" % (len(cells), len(events), len(raw)), flush=True)
        except Exception as exc:
            failures.append(dict(laser=laser, stack=stack_id, error=repr(exc)))
            warnings.warn("Failed %s: %s" % (stack_id, exc))
    pd.DataFrame(failures, columns=["laser", "stack", "error"]).to_csv(
        args.output_dir / "failures.csv", index=False)
    if not all_counts:
        raise RuntimeError("No stacks completed; see failures.csv")
    counts = pd.concat(all_counts, ignore_index=True)
    counts.to_csv(args.output_dir / "cell_timepoints.csv", index=False)
    pd.DataFrame(all_events, columns=EVENT_COLUMNS).to_csv(args.output_dir / "events.csv", index=False)
    pd.DataFrame(stack_info).to_csv(args.output_dir / "stack_summary.csv", index=False)
    stack, summary = summarize(counts, args.frames)
    stack.to_csv(args.output_dir / "stack_frequency.csv", index=False)
    summary.to_csv(args.output_dir / "frequency_by_condition.csv", index=False)
    for cumulative in (False, True):
        name = "flash_cumulative" if cumulative else "flash_frequency"
        plot_statistics(stack, cumulative).to_csv(
            args.output_dir / (name + "_plot_data.csv"), index=False)
    save_plots(summary, args.output_dir)
    print("Saved results to %s" % args.output_dir, flush=True)
    if failures:
        raise RuntimeError("%s stacks failed. Outputs describe successful stacks only; see failures.csv"
                           % len(failures))


if __name__ == "__main__":
    main()



