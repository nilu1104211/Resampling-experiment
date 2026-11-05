"""M1 only: PUFM++-inspired single-step endpoint squared Chamfer refinement.

Source: reference/Enhanced_PUFM-main/train_pufm_s2.py. No RIN or ATS added.
"""
import torch


def build_candidate(config):
    return None


def candidate_loss(prediction, target, config):
    # Same reduction for M1a/M1b; M2 replaces just this loss with upstream APML.
    distance = torch.cdist(prediction, target).square()
    return .5 * (distance.min(-1).values.mean() + distance.min(-2).values.mean())
