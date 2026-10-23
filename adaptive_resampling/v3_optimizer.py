"""Opt-in fused Adam/AdamW updates, including legacy checkpoint resumes."""
import torch


def make_optimizer(optimizer_type, parameters, *, fused=False, **kwargs):
    parameters = list(parameters)
    if fused and any(p.device.type != "cuda" for p in parameters):
        raise ValueError("--fused-optimizer requires CUDA parameters")
    return optimizer_type(parameters, **kwargs, **({"fused": True} if fused else {}))


def load_optimizer_state(optimizer, state, *, fused=False):
    optimizer.load_state_dict(state)
    if not fused:
        return
    # load_state_dict restores param-group flags from the old checkpoint, which
    # would silently turn fused off again. Old serial/foreach step counters also
    # live on CPU; fused Adam requires them on the parameter's CUDA device.
    for group in optimizer.param_groups:
        group["fused"], group["foreach"] = True, False
        for parameter in group["params"]:
            if parameter.device.type != "cuda":
                raise ValueError("--fused-optimizer requires CUDA parameters")
            saved = optimizer.state.get(parameter, {})
            if "step" in saved:
                saved["step"] = saved["step"].to(device=parameter.device, dtype=torch.float32)
