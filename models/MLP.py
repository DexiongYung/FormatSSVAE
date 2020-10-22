import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    """
    Class for standard neural net in PyTorch
    input_size: The input nodes of the neural net
    hidden_size: The size of hidden layer for neural net
    output_size: The output size of the neural net
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, softplus: bool = False):
        super(NeuralNet, self).__init__()
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(input_size, hidden_size * 2)
        self.hidden_2 = nn.Linear(hidden_size * 2, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

        if softplus:
            self.softplus = nn.Softplus()
        else:
            self.softplus = None

        # Define sigmoid activation and softplus output
        self.selu = nn.SELU()

    def forward(self, x: list):
        """
        Forward data through the neural net
        x: Inputted tensor to forward through neural net
        """
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.selu(x)
        x = self.hidden_2(x)
        x = self.selu(x)
        x = self.output(x)

        if self.softplus is not None:
            x = self.softplus(x)

        return x
