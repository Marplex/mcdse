import torch
from torch.nn import CrossEntropyLoss
from matryoshka import MatryoshkaLoss

class DseCrossEntropyLoss(torch.nn.Module):
    def __init__(self, temperature: float = 0.02):
        super().__init__()
        self.ce_loss = CrossEntropyLoss(reduction='mean')
        self.temperature = temperature

    def forward(self, q_reps, p_reps):
        scores = torch.einsum("bd,cd->bc", q_reps, p_reps)
        return self.ce_loss(scores / self.temperature, torch.arange(scores.shape[0], device=scores.device))

class MrlCrossEntropyLoss(torch.nn.Module):
    def __init__(self,
                 matryoshka_dims: list[int],
                 matryoshka_weights: list[float | int] | None = None,
                 temperature: float = 0.02):
        super().__init__()
        self.ce_loss = DseCrossEntropyLoss(temperature)
        self.mrl_loss = MatryoshkaLoss(
            loss=self.ce_loss,
            matryoshka_dims=matryoshka_dims,
            matryoshka_weights=matryoshka_weights
        )

    def forward(self, q_reps, p_reps):
        return self.mrl_loss(q_reps, p_reps)
