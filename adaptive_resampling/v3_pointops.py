"""Deterministic FPS with an optional NVRTC kernel; no extension build/install.

The CUDA path preserves the V3 reference's centroid-farthest start, lowest-index
tie break and exclusion of already selected indices (including duplicate xyz).
Set V3_FPS_BACKEND=torch for the reference or cuda to require acceleration.
"""
import ctypes as ct
from functools import lru_cache
import os
from pathlib import Path
import warnings

import torch


_CUDA_SOURCE = r'''
extern "C" __global__ void v3_fps(
    const float* xyz, const long long* starts, float* distance,
    long long* output, int n, int m, int exclude_selected, float initial_distance, int xy_first) {
  const int b = blockIdx.x, tid = threadIdx.x;
  xyz += (long long)b*n*3;
  distance += (long long)b*n;
  output += (long long)b*m;
  __shared__ float values[256];
  __shared__ int indices[256];
  __shared__ int selected;
  if (tid == 0) selected = (int)starts[b];
  for (int k=tid; k<n; k+=256) distance[k] = initial_distance;
  __syncthreads();
  for (int j=0; j<m; ++j) {
    const int old = selected;
    if (tid == 0) output[j] = old;
    const float x=xyz[old*3], y=xyz[old*3+1], z=xyz[old*3+2];
    float best=-2.0f;
    int besti=n;
    for (int k=tid; k<n; k+=256) {
      const float dx=xyz[k*3]-x, dy=xyz[k*3+1]-y, dz=xyz[k*3+2]-z;
      // Match ATen's reduction order for channel-last/channel-first storage.
      const float d=xy_first ? (dx*dx+dy*dy)+dz*dz : (dx*dx+dz*dz)+dy*dy;
      float v=distance[k];
      if (d<v) v=d;
      if (exclude_selected && k==old) v=-1.0f;
      distance[k]=v;
      if (v>best || (v==best && k<besti)) { best=v; besti=k; }
    }
    values[tid]=best; indices[tid]=besti;
    __syncthreads();
    for (int stride=128; stride>0; stride>>=1) {
      if (tid<stride) {
        const float other=values[tid+stride];
        const int idx=indices[tid+stride];
        if (other>values[tid] || (other==values[tid] && idx<indices[tid])) {
          values[tid]=other; indices[tid]=idx;
        }
      }
      __syncthreads();
    }
    if (tid==0) selected=indices[0];
    __syncthreads();
  }
}
'''


@torch.no_grad()
def fps_indices_reference(points, count, *, start_indices=None, exclude_selected=True, initial_distance=float("inf")):
    batch, size, _ = points.shape
    distance = torch.full((batch, size), initial_distance, device=points.device)
    selected = torch.empty(batch, count, dtype=torch.long, device=points.device)
    farthest = (((points - points.mean(1, keepdim=True)) ** 2).sum(-1).argmax(1)
                if start_indices is None else start_indices)
    rows = torch.arange(batch, device=points.device)
    for i in range(count):
        selected[:, i] = farthest
        center = points[rows, farthest][:, None]
        distance = torch.minimum(distance, ((points - center) ** 2).sum(-1))
        if exclude_selected:
            distance.scatter_(1, farthest[:, None], -1)
        farthest = distance.argmax(1)
    return selected


def _check(code, operation):
    if code:
        raise RuntimeError(f"{operation} failed with CUDA/NVRTC status {code}")


@lru_cache(None)
def _libraries():
    torch_lib = Path(torch.__file__).parent / "lib"
    if os.name == "nt":
        candidates = list(torch_lib.glob("nvrtc64_*.dll"))
        if os.environ.get("CUDA_PATH"):
            candidates += list((Path(os.environ["CUDA_PATH"]) / "bin").glob("nvrtc64_*.dll"))
        driver = ct.WinDLL("nvcuda.dll")
    else:
        candidates = ["libnvrtc.so.12", "libnvrtc.so.11.2", "libnvrtc.so"]
        candidates += list(torch_lib.glob("libnvrtc*.so*"))
        for root in map(Path, __import__("sys").path):
            candidates += list((root / "nvidia/cuda_nvrtc/lib").glob("libnvrtc.so*"))
        driver = ct.CDLL("libcuda.so.1")
    errors = []
    for path in candidates:
        try:
            nvrtc = ct.CDLL(str(path))
            break
        except OSError as error:
            errors.append(str(error))
    else:
        raise RuntimeError("NVRTC library not found in the PyTorch CUDA installation: " + "; ".join(errors[-2:]))
    return driver, nvrtc


@lru_cache(None)
def compile_cuda_source(source, names, device, disable_fma=False):
    driver, nvrtc = _libraries()
    program = ct.c_void_p()
    _check(nvrtc.nvrtcCreateProgram(ct.byref(program), source.encode(), b"v3_pointops.cu", 0, None, None), "nvrtcCreateProgram")
    major, minor = torch.cuda.get_device_capability(device)
    options = [f"--gpu-architecture=compute_{major}{minor}".encode()]
    if disable_fma:
        options.append(b"--fmad=false")
    try:
        result = nvrtc.nvrtcCompileProgram(program, len(options), (ct.c_char_p * len(options))(*options))
        if result:
            size = ct.c_size_t()
            nvrtc.nvrtcGetProgramLogSize(program, ct.byref(size))
            log = ct.create_string_buffer(size.value)
            nvrtc.nvrtcGetProgramLog(program, log)
            raise RuntimeError("V3 CUDA compilation failed: " + log.value.decode(errors="replace"))
        size = ct.c_size_t()
        _check(nvrtc.nvrtcGetPTXSize(program, ct.byref(size)), "nvrtcGetPTXSize")
        ptx = ct.create_string_buffer(size.value)
        _check(nvrtc.nvrtcGetPTX(program, ptx), "nvrtcGetPTX")
    finally:
        nvrtc.nvrtcDestroyProgram(ct.byref(program))
    module = ct.c_void_p()
    _check(driver.cuModuleLoadData(ct.byref(module), ptx), "cuModuleLoadData")
    functions = {}
    for name in names:
        function = ct.c_void_p()
        _check(driver.cuModuleGetFunction(ct.byref(function), module, name.encode()), "cuModuleGetFunction")
        functions[name] = function
    # Retain the module handle for the lifetime of the cached function.
    return driver, module, functions


@lru_cache(None)
def _kernel(device):
    driver, module, functions = compile_cuda_source(_CUDA_SOURCE, ("v3_fps",), device, disable_fma=True)
    return driver, module, functions["v3_fps"]


_failure = None
_last_backend = "not_used"


def fps_backend_status():
    return {"requested": os.environ.get("V3_FPS_BACKEND", "auto"), "last_backend": _last_backend,
            "cuda_failure": _failure}


@torch.no_grad()
def fps_indices(points, count, *, start_indices=None, exclude_selected=True, initial_distance=float("inf")):
    global _failure, _last_backend
    if points.ndim != 3 or points.shape[-1] != 3 or not 0 < count <= points.shape[1] or len(points) == 0:
        raise ValueError(f"FPS needs nonempty B,N,3 and 0 < count <= N, got {count}/{points.shape}")
    if start_indices is not None and (start_indices.shape != (len(points),) or start_indices.dtype != torch.long or start_indices.device != points.device):
        raise ValueError("FPS start_indices must be B int64 indices on the points device")
    if start_indices is not None and torch.any((start_indices < 0) | (start_indices >= points.shape[1])):
        raise ValueError("FPS start index is outside the input cloud")
    backend = os.environ.get("V3_FPS_BACKEND", "auto")
    if backend not in {"auto", "cuda", "torch"}:
        raise ValueError("V3_FPS_BACKEND must be auto, cuda or torch")
    eligible = points.is_cuda and points.dtype == torch.float32
    if backend == "cuda" and not eligible:
        raise ValueError("CUDA FPS requires CUDA float32 points")
    if eligible and backend != "torch" and (_failure is None or backend == "cuda"):
        try:
            with torch.cuda.device(points.device):
                starts = (((points - points.mean(1, keepdim=True)) ** 2).sum(-1).argmax(1)
                          if start_indices is None else start_indices.contiguous())
                xy_first = points.stride(-1) > points.stride(-2)
                points = points.contiguous()
                distance = torch.empty(points.shape[:2], device=points.device, dtype=points.dtype)
                output = torch.empty(len(points), count, device=points.device, dtype=torch.long)
                driver, module, function = _kernel(points.device.index)
                values = [ct.c_void_p(t.data_ptr()) for t in (points, starts, distance, output)]
                values += [ct.c_int(points.shape[1]), ct.c_int(count), ct.c_int(exclude_selected),
                           ct.c_float(initial_distance), ct.c_int(xy_first)]
                params = (ct.c_void_p * len(values))(*[ct.cast(ct.byref(v), ct.c_void_p) for v in values])
                stream = ct.c_void_p(torch.cuda.current_stream(points.device).cuda_stream)
                _check(driver.cuLaunchKernel(function, len(points), 1, 1, 256, 1, 1, 0, stream, params, None), "cuLaunchKernel")
            _last_backend = "cuda_nvrtc"
            return output
        except (OSError, RuntimeError) as error:
            if backend == "cuda":
                raise
            _failure = str(error)
            warnings.warn(f"Fast V3 FPS unavailable; using torch reference. {_failure}", RuntimeWarning)
    _last_backend = "torch_reference"
    return fps_indices_reference(points, count, start_indices=start_indices,
                                 exclude_selected=exclude_selected, initial_distance=initial_distance)
