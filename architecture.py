import torch
from bidgcn_conv import BIDGCN
from radiusedge_conv import DynamicRadiusEdgeConv

from torch_geometric.nn.models import MLP
from bezier import ber_tensorbasis2


class ParameterizationNet(torch.nn.Module):
    def __init__(self):
        super(ParameterizationNet, self).__init__()
        r = 0.1  # Radius for the radius graph

        mlpinputinterior = [6, 64, 64]
        mlpinputboundary = [8, 64, 64]
        mlpsizehidden = [[2 * 64, 64], [2 * 64, 64], [2 * 64, 64], [2 * 64, 64]]
        mlpsizeoutput = [64 * 5, 256, 256, 2]

        numlayers = len(mlpsizehidden) + 1

        self.edgeconvlist = torch.nn.ModuleList()

        # Input Layer
        mlp_int = MLP(mlpinputinterior, act='relu', norm="batch_norm",
                      batch_norm_kwargs={"track_running_stats": False})
        mlp_bdry = MLP(mlpinputboundary, act='relu', norm="batch_norm",
                       batch_norm_kwargs={"track_running_stats": False})
        self.edgeconvlist.append(BIDGCN(nninteriorboundary=mlp_bdry, nninteriorinterior=mlp_int, r=r, aggr="mean"))

        # Hidden layers
        for i in range(numlayers - 1):
            mlp = MLP(mlpsizehidden[i], act='relu', norm="batch_norm",
                      batch_norm_kwargs={"track_running_stats": False})
            self.edgeconvlist.append(DynamicRadiusEdgeConv(mlp, r=r, aggr="mean"))

        # Output Layer
        self.output = MLP(mlpsizeoutput, act='relu', norm="batch_norm",
                          batch_norm_kwargs={"track_running_stats": False})

    def forward(self, data):
        interior = data.interior
        boundary = data.boundary

        batch = data.batch

        intermediate = [self.edgeconvlist[0]((interior, boundary), batch)]

        # For hidden layers
        for layer in self.edgeconvlist[1:]:
            intermediate[-1] = torch.relu(intermediate[-1])
            intermediate.append(layer(intermediate[-1], batch))

        # Concatenate all the values
        concat = torch.cat(intermediate, axis=1)

        # Output layer
        return torch.sigmoid(self.output(concat))


def lossfunction(params, interior, boundary, degree=2):
    combinedParameters = torch.cat([params, boundary[:, 3:]])
    combinedPoints = torch.cat([interior, boundary[:, :3]])

    # Make collocation matrix
    mat = ber_tensorbasis2(degree, combinedParameters)
    mat = mat.reshape(mat.shape[0], -1)
    result = torch.linalg.lstsq(mat, combinedPoints)

    approximation = torch.tensordot(mat, result.solution, 1)
    return torch.nn.functional.mse_loss(approximation, combinedPoints)
