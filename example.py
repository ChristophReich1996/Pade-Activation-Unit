import torch
import matplotlib.pyplot as plt

from pau import PAU, freeze_pau


def main() -> None:
    # Make random input
    input: torch.Tensor = torch.rand(2, 3, 224, 224, requires_grad=True)
    # Init PAU
    pau: PAU = PAU()
    # Get output
    output: torch.Tensor = pau(input)
    # Calc gradients
    output.sum().backward()
    # Freeze PAU weights
    print(list(pau.parameters()))
    print(pau.state_dict())
    pau = freeze_pau(pau)
    print(list(pau.parameters()))
    print(pau.state_dict())
    # Plot PAUs for different initializations
    x = torch.linspace(-4, 4, 1000)
    for shape in ["relu", "leaky_relu_0_01", "leaky_relu_0_2", "leaky_relu_0_25", "leaky_relu_0_3",
                  "leaky_relu_m0_5", "tanh", "swish", "sigmoid"]:
        pau: PAU = PAU(initial_shape=shape)
        y = pau(x)
        plt.plot(x, y)
        plt.title(shape)
        plt.show()


if __name__ == '__main__':
    main()
