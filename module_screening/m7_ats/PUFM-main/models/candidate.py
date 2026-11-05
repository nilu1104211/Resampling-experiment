"""M7: train-only error-density inverse CDF; six times/five Euler intervals.

Adapted from Enhanced_PUFM train_pufm_s3.py:build_entropic_schedule.
"""
import numpy as np


def build_schedule(grid, errors, *, floor=1e-8):
    grid, errors = np.asarray(grid, dtype='f8'), np.asarray(errors, dtype='f8')
    if grid.ndim != 1 or len(grid) < 2 or errors.shape != grid.shape:
        raise ValueError('Need matching one-dimensional calibration grid and error curve')
    if not np.isfinite(grid).all() or not np.isfinite(errors).all() or (errors < 0).any() or floor <= 0:
        raise ValueError('Calibration grid/errors/floor are invalid')
    if grid[0] != 0 or grid[-1] != 1 or (np.diff(grid) <= 0).any():
        raise ValueError('Grid must increase strictly from 0 to 1')
    density = np.maximum(errors, floor)
    cdf = np.r_[0, np.cumsum(.5 * (density[1:] + density[:-1]) * np.diff(grid))]
    cdf /= cdf[-1]
    times = np.interp(np.linspace(0, 1, 6), cdf, grid)
    times[0], times[-1] = 0., 1.
    if (np.diff(times) <= 0).any():
        raise ValueError('Degenerate ATS schedule')
    return times.tolist(), cdf.tolist()


def build_candidate(config):
    return None


def candidate_loss(prediction, target, config):
    raise RuntimeError('M7 never trains a new network')
