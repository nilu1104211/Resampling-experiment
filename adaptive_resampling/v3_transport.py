"""Exact batched transport matching, with optional parallel CPU assignments.

Only independent SciPy solves are threaded. Costs, tie handling, batch order,
and the gradient path through the caller's gather remain unchanged.
"""
from concurrent.futures import ThreadPoolExecutor
import os

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


_workers = 1
_executor = None
_executor_pid = None


def configure_assignment_workers(workers=1):
    """Set matching threads separately from DataLoader workers (default: serial)."""
    global _workers, _executor, _executor_pid
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("assignment workers must be a positive integer")
    if _executor_pid != os.getpid():
        # A DataLoader child must never submit work to its parent's thread pool.
        _executor = None
        _executor_pid = None
    if workers != _workers and _executor is not None:
        _executor.shutdown(wait=True)
        _executor = None
        _executor_pid = None
    _workers = workers


def _solve(cost):
    return linear_sum_assignment(cost)[1]


def batched_assignment_indices(cost):
    """Return the same ordered column indices as sequential SciPy assignments."""
    global _executor, _executor_pid
    cost = np.asarray(cost)
    if cost.ndim != 3 or min(cost.shape) < 1 or cost.shape[1] != cost.shape[2]:
        raise ValueError("Expected a nonempty B x N x N cost array")
    if _workers == 1 or len(cost) == 1:
        return np.stack([_solve(matrix) for matrix in cost])
    if _executor is None or _executor_pid != os.getpid():
        _executor = ThreadPoolExecutor(max_workers=_workers, thread_name_prefix="v3-assignment")
        _executor_pid = os.getpid()
    # map preserves batch order even when the individual solves finish out of order.
    return np.stack(list(_executor.map(_solve, cost)))


@torch.no_grad()
def assignment_indices(source, target):
    """Keep the original squared Euclidean costs and detached discrete matching."""
    if source.shape != target.shape or source.ndim != 3:
        raise ValueError("Transport requires identical B x N x C shapes")
    cost = torch.cdist(source.detach(), target.detach()).square().cpu().numpy()
    return torch.as_tensor(batched_assignment_indices(cost), device=target.device)
