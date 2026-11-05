"""M2 only: the vendored upstream pure-PyTorch APML endpoint loss.

No perturbation, CD auxiliary objective, ATS, or feature branch is enabled.
"""
from .apml_loss import AdaptiveProbabilisticMatchingLoss


def build_candidate(config):
    if not 0 < config['apml_p_min'] < 1:
        raise ValueError('APML minimum probability must be between 0 and 1')
    return None


def candidate_loss(prediction, target, config):
    # Keep upstream SUM over assignments, including its point-count loss scale.
    return AdaptiveProbabilisticMatchingLoss(min_softmax_value=config['apml_p_min'])(prediction, target)
