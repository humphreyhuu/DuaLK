import torch
import torch.nn as nn


class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=1.0, neg_weight=1.0):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, logits, targets):
        # Compute sigmoid
        logits = torch.sigmoid(logits)

        # Calculate the loss for positive and negative samples
        loss_pos = -self.pos_weight * targets * torch.log(logits + 1e-8)
        loss_neg = -self.neg_weight * (1 - targets) * torch.log(1 - logits + 1e-8)

        # Combine positive and negative loss
        loss = loss_pos + loss_neg

        # Average over batch
        loss = loss.mean()

        return loss
