import torch
class FF(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=1, **kwargs):
        """
        Initializes the Feedforward Neural Network.
        :param in_channels: Number of input features.
        :param hidden_channels: List of hidden layer sizes.
        :param out_channels: Number of output features.
        """
        super(FF, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList([torch.nn.Linear(in_features=in_channels, out_features=hidden_channels)])
        for _ in range(num_layers - 2):
            linear = torch.nn.Linear(in_features=hidden_channels, out_features=hidden_channels)
            self.layers.append(linear)
        self.fc = torch.nn.Linear(in_features=hidden_channels, out_features=out_channels)
 
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = torch.relu(x)
        x = self.fc(x)
        return x