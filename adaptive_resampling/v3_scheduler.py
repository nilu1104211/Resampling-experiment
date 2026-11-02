"""Epoch-based OUNet poly decay, matching thsolver's non-warmup recipe."""
import math

from torch.optim.lr_scheduler import LRScheduler


class OUNetPolyLR(LRScheduler):
    """Use lr0 * (1 - completed_epochs / horizon)**power for the next epoch.

    Call step() once after each completed training epoch. Construction leaves
    epoch zero at lr0; the checkpoint stores the LR for the next epoch.
    """

    def __init__(self, optimizer, schedule_epochs, power=.9):
        if schedule_epochs < 1 or not math.isfinite(power) or power <= 0:
            raise ValueError("OUNet poly requires a positive horizon and finite positive power")
        self.schedule_type = "ounet-poly-v1"
        self.schedule_epochs = schedule_epochs
        self.power = power
        super().__init__(optimizer)

    def get_lr(self):
        factor = max(0., 1. - self.last_epoch / self.schedule_epochs) ** self.power
        return [base_lr * factor for base_lr in self.base_lrs]

    def load_state_dict(self, state_dict):
        for key in ("schedule_type", "schedule_epochs", "power", "base_lrs"):
            if state_dict.get(key) != getattr(self, key):
                raise ValueError(f"OUNet poly checkpoint has incompatible {key}")
        super().load_state_dict(state_dict)
