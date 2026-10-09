"""Attribution mapper: residual delta, separate exemplar/query mappers."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDeltaMapper(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.delta = nn.Linear(d_in, d_in, bias=True)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x + self.delta(x), dim=-1)

    def regularization_loss(self) -> torch.Tensor:
        return torch.norm(self.delta.weight, p="fro")


class AttributionModel(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.exemplar_mapper = ResidualDeltaMapper(d_in)
        self.query_mapper    = ResidualDeltaMapper(d_in)

    def forward(self, exemplar: torch.Tensor, query: torch.Tensor):
        return self.exemplar_mapper(exemplar), self.query_mapper(query)

    def map_exemplar(self, x: torch.Tensor) -> torch.Tensor:
        return self.exemplar_mapper(x)

    def map_query(self, x: torch.Tensor) -> torch.Tensor:
        return self.query_mapper(x)

    def orthogonality_loss(self) -> torch.Tensor:
        return (self.exemplar_mapper.regularization_loss() +
                self.query_mapper.regularization_loss())
