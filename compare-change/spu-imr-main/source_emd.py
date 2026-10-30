"""Run the repository's original SPU EMD kernels through NVRTC.

Only the deprecated C++/PyTorch binding is replaced. Kernel bodies and the
source EMDFunction's backward convention are retained. No Hungarian fallback.
"""
import ctypes as ct
from functools import lru_cache
from pathlib import Path
import re

import torch
from adaptive_resampling.v3_pointops import compile_cuda_source, _check


@lru_cache(None)
def kernel_source():
    header = Path(__file__).parent / "eval/emd_pytorch_implement/pkg/include/cuda/emd.cuh"
    text = header.read_text(encoding="utf-8")
    text = re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)
    blocks = []
    for name in ("approxmatch", "matchcost", "matchcostgrad1", "matchcostgrad2"):
        match = re.search(r"template\s*<\s*typename\s+T\s*>\s*__global__\s+void\s+" + name + r"\(", text)
        if match is None:
            raise ValueError(f"Source EMD kernel missing: {name}")
        opening = text.index("{", match.end())
        level, end = 1, opening + 1
        while level:
            level += (text[end] == "{") - (text[end] == "}")
            end += 1
        block = text[match.start():end]
        block = re.sub(r"template\s*<\s*typename\s+T\s*>\s*__global__\s+void\s+" + name,
                       'extern "C" __global__ void spu_' + name, block, count=1)
        blocks.append(re.sub(r"\bT\b", "float", block))
    return "\n".join(blocks)


def launch(name, xyz1, xyz2, match, output):
    names = tuple("spu_" + n for n in ("approxmatch", "matchcost", "matchcostgrad1", "matchcostgrad2"))
    driver, module, functions = compile_cuda_source(kernel_source(), names, xyz1.device.index)
    b, n, _ = xyz1.shape
    m = xyz2.shape[1]
    values = [ct.c_int(b), ct.c_int(n), ct.c_int(m)]
    values += [ct.c_void_p(t.data_ptr()) for t in (xyz1, xyz2, match, output)]
    params = (ct.c_void_p * len(values))(*[ct.cast(ct.byref(v), ct.c_void_p) for v in values])
    grid_y, threads = (32, 256) if name == "matchcostgrad2" else (1, 512)
    stream = ct.c_void_p(torch.cuda.current_stream(xyz1.device).cuda_stream)
    _check(driver.cuLaunchKernel(functions["spu_" + name], 32, grid_y, 1, threads, 1, 1,
                                0, stream, params, None), "SPU " + name)


class SourceEMDFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        if xyz1.shape != xyz2.shape or xyz1.ndim != 3 or xyz1.shape[-1] != 3 or not xyz1.shape[0] or not xyz1.shape[1]:
            raise ValueError("Source EMD expects equal B,N,3 point sets")
        if not xyz1.is_cuda or not xyz2.is_cuda or xyz1.dtype != torch.float32 or xyz2.dtype != torch.float32 or xyz1.device != xyz2.device:
            raise ValueError("Source EMD NVRTC requires CUDA float32 inputs on one device")
        with torch.cuda.device(xyz1.device):
            xyz1, xyz2 = xyz1.contiguous(), xyz2.contiguous()
            b, n, _ = xyz1.shape
            match = torch.zeros((b, n, n), device=xyz1.device, dtype=xyz1.dtype)
            cost = torch.zeros(b, device=xyz1.device, dtype=xyz1.dtype)
            temp = torch.zeros((b, 4 * n), device=xyz1.device, dtype=xyz1.dtype)
            launch("approxmatch", xyz1, xyz2, match, temp)
            launch("matchcost", xyz1, xyz2, match, cost)
            ctx.save_for_backward(xyz1, xyz2, match)
        return cost

    @staticmethod
    def backward(ctx, grad_output):
        xyz1, xyz2, match = ctx.saved_tensors
        with torch.cuda.device(xyz1.device):
            grad1, grad2 = torch.zeros_like(xyz1), torch.zeros_like(xyz2)
            launch("matchcostgrad1", xyz1, xyz2, match, grad1)
            launch("matchcostgrad2", xyz1, xyz2, match, grad2)
        # Match the supplied source wrapper, which does not apply grad_output.
        return grad1, grad2


class EMDLoss(torch.nn.Module):
    def forward(self, xyz1, xyz2):
        return SourceEMDFunction.apply(xyz1, xyz2)
