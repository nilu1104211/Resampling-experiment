"""OUNet V3 octrees, without clipping, and explicit confidence-ranked output budget."""
import math
from types import SimpleNamespace
import torch
from torch import nn
import torch.nn.functional as F
import ocnn
from ocnn.octree import Octree, Points, merge_octrees
from model import OUNet
from v3_common import validate_prediction


class BudgetOUNet(OUNet):
    def decoder(self, convs, octree_in, octree_out, update_octree=False):
        count = getattr(self, "output_budget", None)
        if not update_octree or count is None:
            return super().decoder(convs, octree_in, octree_out, update_octree)
        if octree_in.batch_size != 1:
            raise ValueError("Budgeted OUNet decoding processes one object at a time")
        logits = {}
        deconv = convs[self.full_depth]
        self.budget_trace = []
        for i, d in enumerate(range(self.full_depth, self.depth + 1)):
            if d > self.full_depth:
                deconv = self.upsample[i - 1](deconv, octree_out, d - 1)
                deconv = deconv + ocnn.nn.octree_align(convs[d], octree_in, octree_out, d)
            deconv = self.decoder_blks[i](deconv, octree_out, d)
            logit = self.predict[i][1](self.predict[i][0](deconv, octree_out, depth=d))
            logits[d] = logit
            score = logit[:, 1] - logit[:, 0]
            # Retain enough ancestors to fit M leaves and cap exploratory expansion.
            minimum = math.ceil(count / 8 ** (self.depth - d))
            if minimum > len(score):
                raise RuntimeError("Insufficient octree capacity for the requested output budget")
            if d == self.depth:
                keep = count
            else:
                keep = min(len(score), max(minimum, min(int((score > 0).sum()), self.budget_beam * minimum)))
            split = torch.zeros_like(score, dtype=torch.int32)
            split[score.topk(keep).indices] = 1
            octree_out.octree_split(split, d)
            self.budget_trace.append({"depth": d, "candidates": len(score), "selected": keep})
            if d < self.depth:
                octree_out.octree_grow(d + 1)
            else:
                signal = self.header[1](self.header[0](deconv, octree_out, depth=d))
                signal = ocnn.nn.octree_depad(.5 * torch.tanh(signal), octree_out, d)
                octree_out.features[d] = signal
        return {"logits": logits, "signal": signal, "octree_out": octree_out}


class Adapter(nn.Module):
    compatibility_keys = ("octree_depth", "octree_scale")
    training_description = "Native OUNet occupancy cross-entropy + local displacement regression; common V3 input/clean GT; fixed input-only scale, no clipping"
    inference_description = "OUNet with budget-constrained confidence-ranked leaf splitting; exactly M leaves; ancestor beam disclosed; no repeated-point padding"

    def __init__(self, args):
        super().__init__()
        self.args = args
        if not 4 <= args.octree_depth <= 10 or not 0 < args.octree_scale < 1 or args.octree_beam < 1:
            raise ValueError("Use depth 4..10, scale 0..1 and positive ancestor beam")
        flags = SimpleNamespace(channel_in=4, channel_out=3,
                                channels=[512, 512, 256, 256, 128, 128, 64, 64, 32, 32, 32],
                                depth=args.octree_depth, full_depth=2, group=16, feature="LF",
                                resblk_num=2, bottleneck=2)
        self.model = BudgetOUNet(flags)
        self.model.budget_beam = args.octree_beam

    def build_octree(self, points, *, features):
        scaled = points * self.args.octree_scale
        if not torch.isfinite(scaled).all() or torch.any(scaled.abs() >= 1):
            raise ValueError("OUNet coordinate outside (-1,1); lower --octree-scale consistently for the run. Points are never clipped.")
        trees = []
        for cloud in scaled:
            feature = torch.zeros_like(cloud[:, :1]) if features else None
            tree = Octree(self.args.octree_depth, 2, device=cloud.device)
            tree.build_octree(Points(cloud, features=feature))
            trees.append(tree)
        tree = merge_octrees(trees) if len(trees) > 1 else trees[0]
        tree.construct_all_neigh()
        return tree

    def training_loss(self, points, gt):
        if points.shape[1:] != (256, 3) or gt.shape[1:] != (1024, 3):
            raise ValueError("OUNet V3 training expects 256 -> 1024 patches")
        tree = self.build_octree(points, features=True)
        target = self.build_octree(gt, features=False)
        output = self.model(tree, target, update_octree=False)
        losses = [F.cross_entropy(logits, target.nempty_mask(d).long())
                  for d, logits in output["logits"].items()]
        signal = ocnn.modules.InputFeature("L", nempty=True)(target)
        losses.append((output["signal"] - signal).square().sum(1).mean())
        return torch.stack(losses).sum()

    def predict(self, points, output_count):
        if points.shape[1] * 4 != output_count:
            raise ValueError("V3 OUNet requires exactly 4x output")
        self.model.output_budget = output_count
        results = []
        for cloud in points.split(1):
            tree = self.build_octree(cloud, features=True)
            out = self.model(tree, update_octree=True)["octree_out"]
            x, y, z, _ = out.xyzb(out.depth, nempty=True)
            positions = torch.stack([x, y, z], dim=1) + .5 + out.features[out.depth]
            positions = positions / 2 ** (out.depth - 1) - 1
            results.append(positions[None] / self.args.octree_scale)
        self.last_diagnostics = {"octree_budget_trace": self.model.budget_trace}
        return validate_prediction(torch.cat(results), len(points), output_count)
