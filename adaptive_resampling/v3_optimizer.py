"""Opt-in fused Adam/AdamW updates, including legacy checkpoint resumes."""
import torch


def make_optimizer(optimizer_type, parameters, *, fused=False, **kwargs):
    parameters = list(parameters)
    if fused and any(p.device.type != "cuda" for p in parameters):
        raise ValueError("--fused-optimizer requires CUDA parameters")
    return optimizer_type(parameters, **kwargs, **({"fused": True} if fused else {}))


def load_optimizer_state(optimizer, state, *, fused=False, disable_fused=False):
    if fused and disable_fused:
        raise ValueError("Cannot both enable and disable the fused optimizer")
    optimizer.load_state_dict(state)
    if not fused and not disable_fused:
        return
    # load_state_dict restores param-group flags from the old checkpoint, which
    # would silently turn fused off again. Old serial/foreach step counters also
    # live on CPU; fused Adam requires them on the parameter's CUDA device.
    for group in optimizer.param_groups:
        group["fused"], group["foreach"] = (False, None) if disable_fused else (True, False)
        for parameter in group["params"]:
            if fused and parameter.device.type != "cuda":
                raise ValueError("--fused-optimizer requires CUDA parameters")
            saved = optimizer.state.get(parameter, {})
            if "step" in saved:
                device = parameter.device if fused or group.get("capturable", False) else torch.device("cpu")
                saved["step"] = saved["step"].to(device=device, dtype=torch.float32)
    if disable_fused:
        optimizer.defaults["fused"], optimizer.defaults["foreach"] = False, None
        optimizer._step_supports_amp_scaling = False
