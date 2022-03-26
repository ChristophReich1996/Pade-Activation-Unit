# Padé Activation Units: End-to-end Learning of Flexible Activation Functions in Deep Networks

Unofficial **PyTorch** reimplementation of the
paper [Padé Activation Units: End-to-end Learning of Flexible Activation Functions in Deep Networks](https://openreview.net/pdf?id=BJlBSkHtDS)
by Molina et al. published at ICLR 2020.

**This repository includes a pure and easy to use PyTorch implementation of the Padé Activation Unit (PAU).**

Please note that the [official implementation](https://github.com/ml-research/pau) provides a probably faster CUDA
implementation!

## Installation

The PAU can be installed by using `pip`.

````shell script
pip install git+https://github.com/ChristophReich1996/Pade-Activation-Unit
````

## Example Usage

The PAU can be simply used as a standard `nn.Module`:

````python
import torch
import torch.nn as nn
from pau import PAU

network: nn.Module = nn.Sequential(
    nn.Linear(2, 2),
    PAU(),
    nn.Linear(2, 2)
)

output: torch.Tensor = network(torch.rand(16, 2))
````

The PAU is implemented in an efficient way (checkpointing and sequential computation), if you want to use the faster but
more memory intensive version please use `PAU(efficient=False)`

If a nominator degree of 5 and a denominator degree of 4 is used the following initializations are available:
ReLU (`initial_shape=relu`), Leaky ReLU negative slope=0.01 (`initial_shape=leaky_relu_0_01`), Leaky ReLU negative
slope=0.2 (`initial_shape=leaky_relu_0_2`), Leaky ReLU negative slope=0.25 (`initial_shape=leaky_relu_0_25`), Leaky ReLU
negative slope=0.3 (`initial_shape=leaky_relu_0_3`), Leaky ReLU negative slope=-0.5 (`initial_shape=leaky_relu_m0_5`),
Tanh (`initial_shape=tanh`), Swish (`initial_shape=swish`), Sigmoid (`initial_shape=sigmoid`).

If a different nominator and denominator degree or `initial_shape=None` us utilized the PAU is initialized with random
weights/shape.

If you would like to fix the weights of multiple PAUs in a `nn.Module` just call `module = pau.freeze_pau(module)`.

For a more detailed examples on who to use this implementation please refer to the [example](examples.py) file.

The PAU takes the following parameters.

| Parameter | Description | Type |
| ------------- | ------------- | ------------- |
| m | Size of nominator polynomial. Default 5. | int |
| n | Size of denominator polynomial. Default 4. | int |
| initial_shape | Initial shape of PAU, if None random shape is used, also if m and n are not the default value (5 and 4) a random shape is utilized. Default "leaky_relu_0_2". | Optional[str] |
| efficient | If true efficient variant with checkpointing is used. Default True. | bool |
| eps | Constant for numerical stability. Default 1e-08. | float |
| **kwargs | Unused additional key word arguments | Any |

## Reference

````bibtex
@inproceedings{Molina2020,
    title={{Padé Activation Units: End-to-end Learning of Flexible Activation Functions in Deep Networks}},
    author={Alejandro Molina and Patrick Schramowski and Kristian Kersting},
    booktitle={International Conference on Learning Representations},
    year={2020}
}
````
