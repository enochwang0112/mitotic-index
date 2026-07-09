import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    n_classes = logits.shape[1]
    probs = F.softmax(logits, dim=1)
    target_1h = F.one_hot(target, num_classes=n_classes).permute(0, 3, 1, 2).float()

    dims = (0, 2, 3)
    intersection = (probs * target_1h).sum(dims)
    cardinality = probs.sum(dims) + target_1h.sum(dims)
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice.mean()


class SegLoss(nn.Module):
    def __init__(self, class_weights=(1.0, 1.0, 2.0), ce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.class_weights.to(logits.dtype))
        dice = soft_dice_loss(logits, target)
        return self.ce_weight * ce + self.dice_weight * dice
