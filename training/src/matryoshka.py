from torch import nn, Tensor

class MatryoshkaLoss(nn.Module):
    def __init__(
        self,
        loss: nn.Module,
        matryoshka_dims: list[int],
        matryoshka_weights: list[float | int] | None = None
    ) -> None:
        super().__init__()
        self.loss = loss

        if matryoshka_weights is None:
            matryoshka_weights = [1] * len(matryoshka_dims)

        dims_weights = zip(matryoshka_dims, matryoshka_weights)
        self.matryoshka_dims, self.matryoshka_weights = zip(
            *sorted(dims_weights, key=lambda x: x[0], reverse=True))

    def shrink(self, tensor: Tensor, dimension: int) -> Tensor:
        return tensor[:, :dimension]

    def forward(self, q_reps, p_reps) -> Tensor:
        loss = sum(
            weight * self.loss(self.shrink(q_reps, dim), self.shrink(p_reps, dim))
            for dim, weight in zip(self.matryoshka_dims, self.matryoshka_weights)
        )

        return loss
