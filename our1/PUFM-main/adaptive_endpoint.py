"""Optional training-only route B, using actual V3 surface samples."""
from bisect import bisect_right

import numpy as np
import torch

from adaptive_resampling.v3_dataset import AdaptiveResamplingDataset
from adaptive_resampling.v3_core import seed
from models.adaptive_ops import assign_regions


class EndpointTrainingDataset(AdaptiveResamplingDataset):
    def __getitem__(self, index):
        if not self.training:
            raise ValueError("Adaptive endpoints require the training split")
        epoch, number = index if isinstance(index, tuple) else (self.epoch, index)
        sample = super().__getitem__(index)
        shard = bisect_right(self.ends, number)
        row = self.row_maps[shard][number - (self.ends[shard - 1] if shard else 0)]
        pool = self._handle(self.paths[shard])["pool_xyz"][row]
        rng = np.random.default_rng(seed(sample["base_id"], epoch, "fm_endpoint"))
        pool = pool[rng.permutation(len(pool))]
        sample["endpoint_pool"] = ((pool - sample["prediction_center"]) / sample["prediction_scale"]) @ sample["prediction_rotation"].T
        return sample


@torch.no_grad()
def adaptive_endpoint(state, pool, reference):
    """Use exact integer quotas or fall back to route A for that sample."""
    counts = (state["budget"] + state["regions"]["anchor_counts"]).long()
    region = assign_regions(pool, state["regions"]["centers"])
    endpoints = []
    used = 0
    for labels, surface, quotas, original in zip(region, pool, counts, reference):
        if (labels.bincount(minlength=len(quotas)) < quotas).any():
            endpoints.append(original)
            continue
        pieces = [surface[labels == j][:n] for j, n in enumerate(quotas.tolist()) if n]
        endpoints.append(torch.cat(pieces))
        used += 1
    return torch.stack(endpoints), used
