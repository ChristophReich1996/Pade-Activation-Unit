import torch

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
    pau = freeze_pau(pau)


if __name__ == '__main__':
    main()
