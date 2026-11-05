"""Read RepKPU's packaged kernel, without generation or reference writes.

Seeded NumPy rotation/noise follows the upstream load_kernels fixed=center
path. The resulting kernel is stored in the model checkpoint as kp_pos.
"""
from pathlib import Path
import numpy as np


def load_kernels(radius, num_kpoints, num_kernels, dimension, fixed):
    if (num_kpoints, num_kernels, dimension, fixed) != (15, 1, 3, 'center'):
        raise ValueError('This port vendors the 15-point center kernel only')
    with Path(__file__).with_name('k_015_center.ply').open('rb') as handle:
        lines = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError('Incomplete PLY header')
            lines.append(line)
            if line.strip() == b'end_header':
                break
        if b'format binary_little_endian 1.0' not in b''.join(lines):
            raise ValueError('Unexpected kernel PLY encoding')
        points = np.frombuffer(handle.read(), dtype='<f8').reshape(15, 3).copy()
    while True:
        u, v = np.random.uniform(-1, 1, (2, 3))
        u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
        if abs(u @ v) <= .99:
            break
    v = v - (u @ v) * u
    v /= np.linalg.norm(v)
    rotation = np.stack([u, v, np.cross(u, v)], -1)
    return (radius * points[None] @ rotation + np.random.normal(scale=radius * .01, size=(1, 15, 3))).astype('f4')
