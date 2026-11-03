"""Marginal curves trained from post-generator, equal-budget counterfactuals."""
import torch
from torch import nn
from torch.nn import functional as F


class GainNet(nn.Module):
    def __init__(self, feature_dim=76, form="monotone"):
        super().__init__()
        self.form = form
        self.network = nn.Sequential(nn.Linear(feature_dim, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 3))

    def parameters_for(self, features):
        raw = self.network(features)
        return F.softplus(raw[..., 0]) + 1e-6, 4 + 16 * F.softplus(raw[..., 1]), .1 + F.softplus(raw[..., 2])

    def scores(self, features):
        return self.network(features)[..., 0]

    def batch_gain(self, features, counts, delta):
        """Sum Delta SINGLE-point gains; no missing Delta factor."""
        a, beta, gamma = self.parameters_for(features)
        positions = counts[..., None] + torch.arange(delta, device=features.device, dtype=features.dtype)
        if self.form == "static":
            return a * delta
        return (a[..., None] / (1 + positions / beta[..., None]).pow(gamma[..., None])).sum(-1)

    def transfer(self, features, budget, receiver, donor, delta):
        batch = torch.arange(len(features), device=features.device)
        a = self.batch_gain(features[batch, receiver][:, None], budget[batch, receiver][:, None], delta).squeeze(1)
        b = self.batch_gain(features[batch, donor][:, None], (budget[batch, donor] - delta)[:, None], delta).squeeze(1)
        return a - b
