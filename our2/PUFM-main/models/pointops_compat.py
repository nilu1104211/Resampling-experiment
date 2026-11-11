"""Pinned portable PUFM neighborhood operations, with deterministic V3 FPS."""
import torch
from .adaptive_ops import fps_ids, knn



class PointOps:
    @staticmethod
    def furthestsampling(points, count):
        return fps_ids(points, count)

    @staticmethod
    def knnquery_heap(count, points, query):
        return knn(points, query, count)

    @staticmethod
    def ballquery(radius, count, points, query):
        with torch.no_grad():
            output = []
            for q in query.split(512, 1):
                distance = torch.cdist(q, points)
                ids = torch.arange(points.shape[1], device=points.device).expand_as(distance).clone()
                ids[distance > radius] = points.shape[1]
                ids = ids.sort(-1).values[..., :min(count, points.shape[1])]
                first = ids[..., :1]
                fallback = distance.argmin(-1, keepdim=True)
                first = torch.where(first == points.shape[1], fallback, first)
                ids = torch.where(ids == points.shape[1], first, ids)
                if ids.shape[-1] < count:
                    ids = torch.cat([ids, first.expand(*first.shape[:-1], count - ids.shape[-1])], -1)
                output.append(ids)
            return torch.cat(output, 1).long()


pointops = PointOps()
