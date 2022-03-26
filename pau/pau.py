from typing import List, Tuple, Union, Optional, Any

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

__all__: List[str] = ["PAU", "freeze_pau"]


class PAU(nn.Module):
    """
    This class implements the Pade Activation Unit proposed in:
    https://arxiv.org/pdf/1907.06732.pdf
    """

    def __init__(
            self,
            m: int = 5,
            n: int = 4,
            initial_shape: Optional[str] = "leaky_relu_0_2",
            efficient: bool = True,
            eps: float = 1e-08,
            **kwargs: Any
    ) -> None:
        """
        Constructor method
        :param m (int): Size of nominator polynomial. Default 5.
        :param n (int): Size of denominator polynomial. Default 4.
        :param initial_shape (Optional[str]): Initial shape of PAU, if None random shape is used, also if m and n are
        not the default value (5 and 4) a random shape is utilized. Default "leaky_relu_0_2".
        :param efficient (bool): If true efficient variant with checkpointing is used. Default True.
        :param eps (float): Constant for numerical stability. Default 1e-08.
        :param **kwargs (Any): Unused
        """
        # Call super constructor
        super(PAU, self).__init__()
        # Save parameters
        self.efficient: bool = efficient
        self.m: int = m
        self.n: int = n
        self.eps: float = eps
        # Init weights
        if (m == 5) and (n == 4) and (initial_shape is not None):
            weights_nominator, weights_denominator = _get_initial_weights(initial_shape=initial_shape)
            self.weights_nominator: Union[nn.Parameter, torch.Tensor] = nn.Parameter(weights_nominator.view(1, -1))
            self.weights_denominator: Union[nn.Parameter, torch.Tensor] = nn.Parameter(weights_denominator.view(1, -1))
        else:
            self.weights_nominator: Union[nn.Parameter, torch.Tensor] = nn.Parameter(torch.randn(1, m + 1) * 0.1)
            self.weights_denominator: Union[nn.Parameter, torch.Tensor] = nn.Parameter(torch.randn(1, n) * 0.1)

    def __repr__(self) -> str:
        """
        Returns the printable representational string of the PAU
        :return (str): Printable representational
        """
        return f"PAU(m={self.m}, n={self.n}, efficient={self.efficient})"

    def freeze(self) -> None:
        """
        Function freezes the PAU weights by converting them to fixed model parameters.
        """
        if isinstance(self.weights_nominator, nn.Parameter):
            weights_nominator = self.weights_nominator.data.clone()
            del self.weights_nominator
            self.register_buffer("weights_nominator", weights_nominator)
        if isinstance(self.weights_denominator, nn.Parameter):
            weights_denominator = self.weights_denominator.data.clone()
            del self.weights_denominator
            self.register_buffer("weights_denominator", weights_denominator)

    def _forward(
            self,
            input: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass
        :param input (torch.Tensor): Input tensor of the shape [*]
        :return (torch.Tensor): Output tensor of the shape [*]
        """
        # Save original shape
        shape: Tuple[int, ...] = input.shape
        # Flatten input tensor
        input: torch.Tensor = input.view(-1)
        if self.efficient:
            # Init nominator and denominator
            nominator: torch.Tensor = torch.ones_like(input=input) * self.weights_nominator[..., 0]
            denominator: torch.Tensor = torch.zeros_like(input=input)
            # Compute nominator and denominator iteratively
            for index in range(1, self.m + 1):
                x: torch.Tensor = (input ** index)
                nominator: torch.Tensor = nominator + x * self.weights_nominator[..., index]
                if index < (self.n + 1):
                    denominator: torch.Tensor = denominator + x * self.weights_denominator[..., index - 1]
            denominator: torch.Tensor = denominator + 1.
        else:
            # Get Vandermonde matrix
            vander_matrix: torch.Tensor = torch.vander(x=input, N=self.m + 1, increasing=True)
            # Compute nominator
            nominator: torch.Tensor = (vander_matrix * self.weights_nominator).sum(-1)
            # Compute denominator
            denominator: torch.Tensor = 1. + torch.abs((vander_matrix[:, 1:self.n + 1]
                                                        * self.weights_denominator).sum(-1))
        # Compute output and reshape
        output: torch.Tensor = (nominator / denominator.clamp(min=self.eps)).view(shape)
        return output

    def forward(
            self,
            input: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass
        :param input (torch.Tensor): Input tensor of the shape [batch size, *]
        :return (torch.Tensor): Output tensor of the shape [batch size, *]
        """
        # Make input contiguous if needed
        input: torch.Tensor = input if input.is_contiguous() else input.contiguous()
        if self.efficient:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input=input)


def freeze_pau(
        module: Union[nn.Module, nn.DataParallel, nn.parallel.DataParallel]
) -> Union[nn.Module, nn.DataParallel, nn.parallel.DataParallel]:
    """
    Function freezes any PAU in a given nn.Module (nn.DataParallel, nn.parallel.DataParallel).
    Function inspired by Timms function to freeze batch normalization layers.
    :param module (Union[nn.Module, nn.DataParallel, nn.parallel.DataParallel]): Original model with frozen PAU
    """
    res = module
    if isinstance(module, PAU):
        res.freeze()
    else:
        for name, child in module.named_children():
            new_child = freeze_pau(child)
            if new_child is not child:
                res.add_module(name, new_child)
    return res


def _get_initial_weights(
        initial_shape: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Function returns the PAU weights corresponding to the initial shape.
    Values taken from Table 6 of the paper.
    :param initial_shape (str): Initial shape of the PAU
    :return (Tuple[torch.Tensor, torch.Tensor]): Numerator (shape is [6]) and denominator (shape is [4]) weights
    """
    # Check input
    assert initial_shape in ["relu", "leaky_relu_0_01", "leaky_relu_0_2", "leaky_relu_0_25", "leaky_relu_0_3",
                             "leaky_relu_m0_5", "tanh", "swish", "sigmoid"], \
        f"Initial shape {initial_shape} is not available."
    # Get weights
    if initial_shape == "relu":
        return torch.tensor((0.02996348,
                             0.61690165,
                             2.37539147,
                             3.06608078,
                             1.52474449,
                             0.25281987), dtype=torch.float), \
               torch.tensor((1.19160814,
                             4.40811795,
                             0.91111034,
                             0.34885983), dtype=torch.float)
    elif initial_shape == "leaky_relu_0_01":
        return torch.tensor((0.02979246,
                             0.61837738,
                             2.32335207,
                             3.05202660,
                             1.48548002,
                             0.25103717), dtype=torch.float), \
               torch.tensor((1.14201226,
                             4.39322834,
                             0.87154450,
                             0.34720652), dtype=torch.float)
    elif initial_shape == "leaky_relu_0_2":
        return torch.tensor((0.02557776,
                             0.66182815,
                             1.58182975,
                             2.94478759,
                             0.95287794,
                             0.23319681), dtype=torch.float), \
               torch.tensor((0.50962605,
                             4.18376890,
                             0.37832090,
                             0.32407314), dtype=torch.float)
    elif initial_shape == "leaky_relu_0_25":
        return torch.tensor((0.02423485,
                             0.67709718,
                             1.43858363,
                             2.95497990,
                             0.85679722,
                             0.23229612), dtype=torch.float), \
               torch.tensor((0.41014746,
                             4.14691964,
                             0.30292546,
                             0.32002850), dtype=torch.float)
    elif initial_shape == "leaky_relu_0_3":
        return torch.tensor((0.02282366,
                             0.69358438,
                             1.30847432,
                             2.97681599,
                             0.77165297,
                             0.23252265), dtype=torch.float), \
               torch.tensor((0.32849543,
                             4.11557902,
                             0.24155603,
                             0.31659365), dtype=torch.float)
    elif initial_shape == "leaky_relu_m0_5":
        return torch.tensor((0.02650441,
                             0.80772912,
                             13.56611639,
                             7.00217900,
                             11.61477781,
                             0.68720375), dtype=torch.float), \
               torch.tensor((13.70648993,
                             6.07781733,
                             12.32535229,
                             0.54006880), dtype=torch.float)
    elif initial_shape == "tanh":
        return torch.tensor((0.,
                             1.,
                             0.,
                             1. / 9.,
                             0.,
                             1. / 945), dtype=torch.float), \
               torch.tensor((0.,
                             4. / 9.,
                             0.,
                             1. / 63.), dtype=torch.float)
    elif initial_shape == "swish":
        return torch.tensor((0.,
                             1. / 2.,
                             1. / 4.,
                             3. / 56.,
                             1. / 168.,
                             1. / 3360.), dtype=torch.float), \
               torch.tensor((0.,
                             1. / 28.,
                             0.,
                             1. / 1680.), dtype=torch.float)
    elif initial_shape == "sigmoid":
        return torch.tensor((1. / 2.,
                             1. / 4.,
                             1. / 18.,
                             1. / 144.,
                             1. / 2016.,
                             1. / 60480.), dtype=torch.float), \
               torch.tensor((0.,
                             1. / 9.,
                             0.,
                             1. / 10008.), dtype=torch.float)
    else:
        raise ValueError(f"Initial shape {initial_shape} not found!")
