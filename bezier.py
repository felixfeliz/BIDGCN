import torch
import math
from typing import Iterator, Optional
import numpy as np
from scipy.special import binom
from torch.utils.data import IterableDataset
import random

from scipy.spatial.transform import Rotation

def greville_abscissa(knots, d):
    # knots : [bs, num_of_knots]
    k = d + 1
    tstar = torch.zeros(knots.shape[0], knots.shape[1] - k)
    for i in range(knots.shape[1] - k):
        tstar[:, i] = torch.mean(knots[:, i + 1:i + k], 1)
    return tstar

def scale_dim_points(p):
    # input points p have dimension [bs, num_of_points, 3]
    dim = p.shape[-1]
    min_p = []
    max_p = []
    for i in range(dim):
        min_p.append(torch.min(p[:, i]))
    for i in range(dim):
        max_p.append(p[:, i].max() - min_p[i])
    max_p = max(max_p)
    min_p = torch.FloatTensor(min_p).to(p.device).unsqueeze(0)
    p1 = (p - min_p) / max_p
    return p1

def ber_tensorbasis2(d, params):
    binomial = binom(d, np.arange(0,d+1))

    basisU = torch.empty(params.shape[0], d+1, device=params.device)
    basisV = torch.empty(params.shape[0], d+1, device=params.device)

    for j in range(d+1):
        basisU[:, j] = binomial[j]*params[:, 0].pow(j) * (1 - params[:, 0]).pow(d - j)
        basisV[:, j] = binomial[j]*params[:, 1].pow(j) * (1 - params[:, 1]).pow(d - j)

    result = torch.matmul(basisU.reshape(-1, d+1, 1), basisV.reshape(-1, 1, d+1))
    return result


class BezierSurf:
    def __init__(self, d: int, device: str):
        self.d = d
        self.device = device
        self.knots = torch.cat((torch.zeros(self.d + 1), torch.ones(self.d + 1)))
        self.gx = greville_abscissa(self.knots.unsqueeze(0), self.d)[0].to(self.device)
        self.coefs = torch.zeros(((self.d + 1)**2, 3))

    def eval(self, params):
        # Eval basis
        mat = ber_tensorbasis2(self.d, torch.tensor(params).transpose(1,0))
        mat = mat.reshape(mat.shape[0], -1)
        result = torch.tensordot(mat, self.coefs, dims=1)
        return result.transpose(1,0)

    def rotate(self, phi, rotaxis):
        rotaxis = rotaxis / np.linalg.norm(rotaxis) * phi
        r = Rotation.from_rotvec(rotaxis)
        self.coefs = torch.tensor(r.apply(self.coefs))


class BezierRandomSurface(BezierSurf):
    def __init__(self, d, device):
        super().__init__(d, device)

    def get_random_surface(self):
        surf = BezierSurf(self.d, self.device)  # initialization

        meshx, meshy = torch.meshgrid(surf.gx, surf.gx, indexing='ij')

        meshz = 1 - 2 * torch.rand(meshx.shape)
        perturbationSize = 1 / (2 * self.d)

        meshx = meshx + perturbationSize * (1 - 2 * torch.rand(meshx.shape))
        meshy = meshy + perturbationSize * (1 - 2 * torch.rand(meshx.shape))

        coefs = torch.empty((surf.d + 1) ** 2, 3)
        coefs[:, 0] = meshx.flatten()
        coefs[:, 1] = meshy.flatten()
        coefs[:, 2] = meshz.flatten()

        #surf = BezierSurf.get_surface(surf, coefs=coefs)
        # set coefs
        surf.coefs = coefs

        phi = random.randint(0, 180) * math.pi / 180
        rotaxis = np.random.rand(3)

        surf.rotate(phi, rotaxis)
        return coefs, phi, rotaxis, surf


def generate_scattered_data(d: int, n: int, b: int, device: str):
    coefs, phi, rotaxis, surf = BezierRandomSurface(d, device).get_random_surface()

    # generation of parameters on the boundary isolines, for u = 0,1 and v=0,1
    # according to random uniform distribution
    # assumption: there are b points along each boundary isoline, hence in total 4*b boundary samples
    upar = np.random.random(size=b - 2)
    while np.any(upar == 0.0):
        upar = np.random.random(size=b - 2)
    upar.sort()
    upar = np.concatenate((np.array([0]), upar, np.array([1])))
    dB1 = np.concatenate((upar[0:-1].reshape(1, b - 1), np.zeros(b - 1).reshape(1, b - 1)), axis=0)
    labels = np.concatenate((np.array([5]), np.ones(b - 2)))

    upar = np.random.random(size=b - 2)
    while np.any(upar == 0.0):
        upar = np.random.random(size=b - 2)
    upar.sort()
    upar = np.concatenate((np.array([0]), upar, np.array([1])))
    dB4 = np.concatenate((np.ones(b - 1).reshape(1, b - 1), upar[0:-1].reshape(1, b - 1)), axis=0)
    labels = np.concatenate((labels, np.array([6]), 4 * np.ones(b - 2)))

    upar = np.random.random(size=b - 2)
    while np.any(upar == 0.0):
        upar = np.random.random(size=b - 2)
    upar.sort()
    upar = np.concatenate((np.array([0]), upar, np.array([1])))
    dB2 = np.flip(np.concatenate((upar.reshape(1, b), np.ones(b).reshape(1, b)), axis=0), 1)
    labels = np.concatenate((labels, np.array([7]), 2 * np.ones(b - 2), np.array([8])))

    upar = np.random.random(size=b - 2)
    while np.any(upar == 0.0):
        upar = np.random.random(size=b - 2)
    upar.sort()
    upar = np.concatenate((np.array([0]), upar, np.array([1])))
    dB3 = np.flip(np.concatenate((np.zeros(b - 2).reshape(1, b - 2), upar[1:-1].reshape(1, b - 2)), axis=0), 1)
    labels = np.concatenate((labels, 3 * np.ones(b - 2)))

    PB = np.concatenate((dB1, dB4, dB2, dB3), axis=1)

    # generation of interior parameters
    PI = np.random.random(size=(2, n))
    labels = np.concatenate((np.zeros(n), labels))
    # plt.plot(interior_params[:, 0], interior_params[:, 1], '.b')

    # generation of surfaces scattered datapoints
    # boundary values
    XB = surf.eval(PB)

    # interior values
    XI = surf.eval(PI)

    params = np.concatenate((PI, PB), axis=1).transpose()

    svals = np.concatenate((XI, XB), axis=1)

    assert labels.shape[0] == len(params)
    return coefs, phi, rotaxis, svals, params, labels


class BezierScatteredSurfaceData(IterableDataset):

    def __iter__(self) -> Iterator[dict]:
        while True:
            yield self.sample_bezier_surface()

    def __init__(self, d: int, n: int, device: str, random_noise: Optional[float] = None):

        self.d = d
        self.n = n  # number of interior points

        self.b = math.ceil(math.sqrt(n)) + 2  # number of boundary points
        self.m = n + self.b + 4  # dimension of the point-cloud: m =  n + b + number of patch corner (4 for square
        # domain)
        self.device = device
        self.random_noise = random_noise

    def sample_bezier_surface(self):
        c, phi, rotaxis, vals, params, labels = generate_scattered_data(self.d, self.n, self.b,
                                                                        self.device)

        p = torch.tensor(vals)
        paramTensor = torch.tensor(params)
        labelsTensor = torch.tensor(labels)

        result = dict()
        if self.random_noise is not None:
            p_noised = p + (torch.randn_like(p) * self.random_noise)
            p_noised_scaled = scale_dim_points(p_noised.transpose(0, 1))
            result['p_noised'] = p_noised_scaled.transpose(0, 1)
        p = scale_dim_points(p.transpose(0, 1))
        result['p'] = p.transpose(0, 1)
        result['c'] = c
        result['phi'] = phi
        result['rotaxis'] = torch.tensor(rotaxis)
        result['params'] = paramTensor
        result['labels'] = labelsTensor

        return result

    def __len__(self):
        return self.m



