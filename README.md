# BIDGCN

This is the implementation of the boundary informed dynamic graph convolutional network (BIDGCN)
proposed in

[1] Carlotta Giannelli, Sofia Imperatore, Angelos Mantzaflaris, Felix Scholz. BIDGCN: Boundary informed dynamic graph 
convolutional network for adaptive spline fitting of scattered data. 2023.
[hal-04313629](https://hal.science/hal-04313629).

The key property of this network architecture for learning on point clouds is a novel input layer 
that handles boundary conditions defined on a subset of the input point cloud. 
In [1], the network is applied to the problem of parameterizing scattered 3D point clouds
over a planar domain for adaptive fitting with hierarchical splines. 
This repository also contains the synthetic data set used for training and testing in that publication.
Besides point cloud parameterization, the new input layer can potentially be applied to any other problem where boundary conditions are provided.

## Notes
The implementation use [PyTorch](https://pytorch.org/) and [PyG](https://pyg.org/) and can thus be easily
integrated into existing codes. 

[bidgcn_conv.py](bidgcn_conv.py) contains the novel input layer, which is
implemented as a subclass of PyG's `MessagePassing` module.  

[architecture.py](architecture.py) contains the network architecture proposed in [1].

[dataset.py](dataset.py) contains the code for generating the synthetic training dataset used in [1], 
it uses the helper functions and classes provided in [bezier.py](bezier.py).

[radiusedge_conv.py](radiusedge_conv.py) is equal to the PyG-implementation of the dynamic edge convolution operator by
[Wang et al., 2019](https://doi.org/10.1145/3326362)
with the difference that a radius graph is used instead of a k-nearest neighbor graph.

[training.py](training.py) contains the training loop for reproducing the network in [1].





