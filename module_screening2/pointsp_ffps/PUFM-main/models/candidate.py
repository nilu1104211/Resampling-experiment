"""Fixed-pool PUFM adaptations of PointSP and Denoise Yourself sampling."""
import torch
from torch import nn
from .adaptive_ops import gather, knn
from .adapter_ops import (midpoint_pool, pointsp_weight, geometry, weighted_ids,
                          select_fps, generator, median_value)


class Candidate(nn.Module):
    kind = 'initialization'

    def __init__(self, config):
        super().__init__()
        self.config = config

    def initialize(self, points, original):
        variant = self.config['variant']
        n, count = points.shape[1], original.shape[1]
        pool = midpoint_pool(points)
        diag = {'candidate_count': pool.shape[1], 'selected_count': count}
        if variant == 'uniform_random':
            ids = weighted_ids(pool, pool.new_ones(pool.shape[:2]), count, self.config['seed'])
            return gather(pool, ids), diag
        if variant.startswith('pointsp_') and variant != 'pointsp_lgp':
            weights, neighbors = pointsp_weight(points, self.config['neighbors'])
            # Scores on observed points propagate to each midpoint's two ends.
            mid_neighbors = knn(points, points, 8)
            scores = .5 * (weights[:, :, None] + gather(weights[..., None], mid_neighbors).squeeze(-1))
            scores = scores.flatten(1)
            if variant == 'pointsp_wrs':
                ids = weighted_ids(pool, scores, count, self.config['seed'])
                selected = gather(pool, ids)
            else:
                # Original FFPS is filtering + ordinary FPS, not d*weight FPS.
                keep = max(count, int(pool.shape[1] * (1 - self.config['filter_fraction'])))
                ids = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :keep]
                selected = select_fps(gather(pool, ids), count)
            diag.update(min_weight=scores.min().detach(), max_weight=scores.max().detach())
            return selected, diag
        ids, distance, normal, spacing = geometry(points, self.config['neighbors'])
        if variant.startswith('dy_density'):
            radius = distance[..., -1].sqrt().clamp_min(1e-5)
            log_density = -3 * radius.log()  # k is constant; grouping is unchanged.
            low = log_density < median_value(log_density, keepdim=True)
            favor = low if variant == 'dy_density_low' else ~low
            weights = torch.where(favor, points.new_tensor(2.), points.new_tensor(1.))
            # DY's density groups are adapted to a fixed midpoint-selection budget.
            scores = weights.repeat_interleave(8, 1)
            selected = gather(pool, weighted_ids(pool, scores, count, self.config['seed']))
            diag['favored_parent_fraction'] = favor.float().mean().detach()
            return selected, diag
        if variant == 'dy_curvature':
            cosine = (gather(normal, ids) * normal[:, :, None]).sum(-1).abs().clamp(0, 1)
            angle = cosine.acos().mean(-1)
            weights = 1 + angle / angle.mean(-1, keepdim=True).clamp_min(1e-6)
            selected = gather(pool, weighted_ids(pool, weights.repeat_interleave(8, 1), count,
                                                 self.config['seed']))
            diag['normal_variation'] = angle.mean().detach()
            return selected, diag
        if variant != 'pointsp_lgp':
            raise ValueError(f'Unknown sampler {variant}')
        # PointSP LGP's neighbor vector projected into the local tangent plane.
        proposals = []
        for cloud, neighbor_ids, normals, local_distance in zip(points, ids, normal, distance):
            rng = generator(cloud, self.config['seed'], 'lgp')
            choice = torch.randint(1, neighbor_ids.shape[-1], (n, 8), generator=rng).to(points.device)
            selected_ids = neighbor_ids.gather(1, choice)
            vector = cloud[selected_ids] - cloud[:, None]
            tangent = vector - (vector * normals[:, None]).sum(-1, keepdim=True) * normals[:, None]
            length = tangent.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            limit = median_value(local_distance.sqrt())[:, None, None]
            step = torch.minimum(.5 * length, limit)
            proposals.append((cloud[:, None] + tangent / length * step).flatten(0, 1))
        selected = select_fps(torch.stack(proposals), count)
        diag['normal_source'] = 'observed_xyz_PCA_only'
        return selected, diag


def build_candidate(config):
    return None if config['variant'] in {'b0', 'b_extra'} else Candidate(config)
