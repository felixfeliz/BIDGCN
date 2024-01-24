import torch
from torch_geometric.data import Data, InMemoryDataset
import itertools
from bezier import BezierScatteredSurfaceData


class TensorProductPolynomialData(InMemoryDataset):
    def __init__(self, root, length, d, n=1000, transform=None, pre_transform=None,
                 pre_filter=None, random_noise=False):
        self.length = length
        self.d = d
        self.n = n
        self.random_noise = random_noise

        self.numUknt = 0
        self.numVknt = 0
        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f"data{self.length}degree{self.d}n{self.n}noise{self.random_noise}.pt"]

    def process(self):
        standard_data = BezierScatteredSurfaceData(d=self.d, n=self.n, device='cpu',
                                                  random_noise=self.random_noise)

        data_list = []
        for surface in itertools.islice(standard_data, self.length):
            points = torch.transpose(surface['p' if self.random_noise is None else 'p_noised'], 0, 1)
            params = surface['params']
            interior = points[:self.n]
            boundary = points[self.n:]
            boundaryparams = params[self.n:]
            # For boundary features, add the parameters as extra features
            boundaryfeatures = torch.cat((boundary, boundaryparams), dim=1)

            data_list.append(Data(interior=interior,
                                  boundary=boundaryfeatures, yinterior=params[:self.n], yboundary=boundaryparams,
                                  ref=surface['labels'][:self.n]))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
