import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.typing import OptTensor, PairOptTensor, PairTensor
from torch_cluster import radius

from typing import Callable, Optional, Union


class BIDGCN(MessagePassing):
    def __init__(self, nninteriorboundary: Callable, nninteriorinterior: Callable, r: float, aggr: str = 'max',
                 num_workers: int = 1, **kwargs):
        super().__init__(aggr=aggr, flow='source_to_target', **kwargs)

        if radius is None:
            raise ImportError('`BIDGCN` requires `torch-cluster`.')

        self.nnInteriorBoundary = nninteriorboundary
        self.nnInteriorInterior = nninteriorinterior
        self.r = r
        self.num_workers = num_workers
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nnInteriorInterior)
        reset(self.nnInteriorBoundary)

    def forward(
            self, x: PairTensor,
            batch: Union[OptTensor, Optional[PairTensor]] = None) -> Tensor:

        if x[0].dim() != 2:
            raise ValueError("Static graphs not supported in BIDGCN")

        b: PairOptTensor = (None, None)
        if isinstance(batch, Tensor):
            raise ValueError("Needs to be a tuple.")
        elif isinstance(batch, tuple):
            assert batch is not None
            b = (batch[0], batch[1])

        # Compute 2 radius graphs:
        edge_indexinterior = radius(x=x[0], y=x[0], r=self.r, batch_x=b[0], batch_y=b[0]).flip([0])
        edge_indexboundary = radius(x=x[0], y=x[1][:, :3], r=self.r, batch_x=b[0], batch_y=b[0])

        # propagate_type: (x: PairTensor)
        resultinterior = self.propagate(edge_indexinterior, x=(x[0], x[0]), size=None)
        resultboundary = self.propagate(edge_indexboundary, x=(x[1], x[0]), size=(x[1].shape[0], x[0].shape[0]))

        return resultinterior + resultboundary

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        # Choose nn depending on feature shape (R3->interior, R5->boundary)
        if x_j.shape[1] == 3:
            return self.nnInteriorInterior(torch.cat([x_i, x_j - x_i], dim=-1))
        elif x_j.shape[1] == 5:
            return self.nnInteriorBoundary(torch.cat([x_i, x_j[:, :3] - x_i, x_j[:, 3:]], dim=-1))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn}, k={self.k})'
