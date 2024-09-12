import torch 
from torch import nn
from torch.nn import functional as F

class SoftCrossEntropyWithWeightsLoss(nn.Module):
    def __init__(self, weights):
        super(SoftCrossEntropyWithWeightsLoss, self).__init__()
        self.weights = nn.Parameter(weights, requires_grad=False)

    def forward(self, y_hat, y):
        weighted_logits = F.log_softmax(y_hat, dim=-1) 
        # The choice of dim is important.
        # Remember, we need to work on N x D matrix and thus, w1 x11 + w2 x12 +...wN x1N / w1 + ...wN => dim=0
        # for the summation
        weighted_sum = torch.sum(-y * weighted_logits, dim=0) / self.weights.sum()
        return weighted_sum.mean()

    def __repr__(self):
        return f"weights are on {self.weights.device}\n" 