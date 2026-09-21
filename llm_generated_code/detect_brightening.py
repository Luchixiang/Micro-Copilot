"""Detect sudden brightening in a TIFF with axes (time, y, x).

Dependencies: pip install numpy tifffile
Example:
    python detect_brightening.py movie.tif --window 16 16 --stride 8 8

Each window is compared with its preceding frames (current frame excluded).
A detection requires BOTH a baseline excursion and a frame-to-frame jump.
All coordinates and frame indices are zero-based; window ends are exclusive.
This detects brightening, not its biological cause: motion or illumination
changes can also trigger detections. Register moving images first if needed.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import tifffile


def window_means(images, window=(16, 16), stride=(8, 8)):
    """Return means (T, Ny, Nx) and spatial window origins (ys, xs).

    Include a final full window at each image edge when stride does not divide
    the image evenly. Integral images avoid materializing pixel patches.
    """
    images = np.asarray(images)
    if images.ndim != 3 or not np.issubdtype(images.dtype, np.number):
        raise ValueError('Expected a numeric TIFF array with shape (T, Y, X).')
    if np.iscomplexobj(images) or not np.isfinite(images).all():
        raise ValueError('Images must contain finite, real intensities.')
    if len(window) != 2 or len(stride) != 2:
        raise ValueError('window and stride must each contain two integers.')
    if any(int(v) != v or v < 1 for v in (*window, *stride)):
        raise ValueError('Window and stride values must be positive integers.')
    h, w = map(int, window)
    sy, sx = map(int, stride)
    t, height, width = images.shape
    if t < 4 or h > height or w > width:
        raise ValueError('Need >=4 frames and windows no larger than the image.')
    if sy > h or sx > w:
        raise ValueError('Stride must not exceed window size (would leave gaps).')
    ys = np.unique(np.r_[np.arange(0, height - h + 1, sy), height - h])
    xs = np.unique(np.r_[np.arange(0, width - w + 1, sx), width - w])
    means = np.empty((t, len(ys), len(xs)), dtype=np.float64)
    y0, x0 = ys[:, None], xs[None, :]
    for frame in range(t):
        integral = np.pad(images[frame].astype(np.float64), ((1, 0), (1, 0)))
        integral.cumsum(axis=0, out=integral)
        integral.cumsum(axis=1, out=integral)
        means[frame] = (integral[y0 + h, x0 + w] - integral[y0, x0 + w]
                       - integral[y0 + h, x0] + integral[y0, x0]) / (h * w)
    return means, ys, xs


def detect_brightening(images, window=(16, 16), stride=(8, 8),
                       baseline_frames=10, min_history=5, z_threshold=5.0,
                       min_relative=0.2, min_increase=0.0, noise_floor=1.0):
    """Return window-level measurements and a boolean detection array.

    baseline = median of the preceding baseline_frames window means
    sigma = max(1.4826 * median(abs(history - baseline)), noise_floor)
    delta = current mean - baseline
    jump = current mean - previous mean
    required = max(z_threshold*sigma, min_relative*abs(baseline), min_increase)
    detection = (delta > required) AND (jump > required)

    The jump requirement selects abrupt changes rather than sustained elevated
    intensity or slow ramps. z_score is a robust effect score, not a p-value.
    noise_floor and min_increase are in input intensity units, averaged per
    window; adjust noise_floor for normalized images (e.g. 0.001 for 0..1 data).
    The first min_history frames have no baseline and cannot be detections.
    Overlapping windows may report the same event more than once.
    """
    if not 3 <= min_history <= baseline_frames:
        raise ValueError('Require 3 <= min_history <= baseline_frames.')
    if any(not np.isfinite(v) or v < 0 for v in
           (z_threshold, min_relative, min_increase, noise_floor)) or noise_floor == 0:
        raise ValueError('Thresholds must be finite/nonnegative; noise_floor > 0.')
    means, ys, xs = window_means(images, window, stride)
    if means.shape[0] <= min_history:
        raise ValueError('Number of frames must exceed min_history.')
    baseline = np.full_like(means, np.nan)
    sigma = np.full_like(means, np.nan)
    for t in range(min_history, len(means)):
        history = means[max(0, t - baseline_frames):t]
        baseline[t] = np.median(history, axis=0)
        sigma[t] = np.maximum(
            1.4826 * np.median(np.abs(history - baseline[t]), axis=0), noise_floor)
    delta = means - baseline
    jump = np.full_like(means, np.nan)
    jump[1:] = np.diff(means, axis=0)
    relative = delta / np.maximum(np.abs(baseline), noise_floor)
    required = np.maximum(np.maximum(z_threshold * sigma,
                                     min_relative * np.abs(baseline)), min_increase)
    detected = (delta > required) & (jump > required)
    return dict(mean=means, baseline=baseline, delta=delta, jump=jump,
                relative_change=relative, z_score=delta / sigma,
                threshold=required, detected=detected, ys=ys, xs=xs,
                window=np.asarray(window), stride=np.asarray(stride))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('input', type=Path)
    p.add_argument('--output', type=Path, default=Path('brightening_results'))
    p.add_argument('--window', type=int, nargs=2, default=(16, 16), metavar=('H', 'W'))
    p.add_argument('--stride', type=int, nargs=2, default=(8, 8), metavar=('Y', 'X'))
    p.add_argument('--baseline-frames', type=int, default=10)
    p.add_argument('--min-history', type=int, default=5)
    p.add_argument('--z-threshold', type=float, default=5.0)
    p.add_argument('--min-relative', type=float, default=0.2)
    p.add_argument('--min-increase', type=float, default=0.0)
    p.add_argument('--noise-floor', type=float, default=1.0)
    p.add_argument('--save-mask', action='store_true', help='Save full TYX event mask.')
    args = p.parse_args()
    images = tifffile.imread(args.input)
    result = detect_brightening(images, **{k: getattr(args, k) for k in
        ('window', 'stride', 'baseline_frames', 'min_history', 'z_threshold',
         'min_relative', 'min_increase', 'noise_floor')})
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / 'window_measurements.npz', **result)
    indices = np.argwhere(result['detected'])
    metrics = ['mean', 'baseline', 'delta', 'jump', 'relative_change', 'z_score']
    h, w = args.window
    with (args.output / 'events.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'y_start', 'y_end', 'x_start', 'x_end', *metrics])
        for t, iy, ix in indices:
            y, x = result['ys'][iy], result['xs'][ix]
            writer.writerow([t, y, y + h, x, x + w,
                             *[result[k][t, iy, ix] for k in metrics]])
    if args.save_mask:
        mask = np.zeros(images.shape, dtype=np.uint8)
        for t, iy, ix in indices:
            y, x = result['ys'][iy], result['xs'][ix]
            mask[t, y:y + h, x:x + w] = 255
        tifffile.imwrite(args.output / 'event_mask.tif', mask,
                         photometric='minisblack', metadata={'axes': 'TYX'})
    print(f'Detected {len(indices)} window events in '
          f'{len(np.unique(indices[:, 0]))} frames. Results: {args.output.resolve()}')


if __name__ == '__main__':
    main()
