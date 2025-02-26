import torch
import torch.nn as nn

class LinearNetwork(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_sizes=[128,128]):
        super(LinearNetwork, self).__init__()

        network_layers = [nn.Linear(input_dims, hidden_sizes[0])]
        for i in range(len(hidden_sizes)-1):
            network_layers.append(
                nn.ReLU(),
            )
            network_layers.append(
                nn.Linear(hidden_sizes[i], hidden_sizes[i+1])
            )
        network_layers.append(
            nn.ReLU()
        )
        network_layers.append(
            nn.Linear(hidden_sizes[-1], output_dims)
        )

        self.layers = nn.Sequential(*network_layers)

    def forward(self, x):
        return self.layers(x)
    

class DuelingNetwork(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_sizes=[128,128]):
        super(DuelingNetwork, self).__init__()


        feature_layers = [nn.Linear(input_dims, hidden_sizes[0])]
        for i in range(len(hidden_sizes)-2):
            feature_layers.append(
                nn.ReLU(),
            )
            feature_layers.append(
                nn.Linear(hidden_sizes[i], hidden_sizes[i+1])
            )
        feature_layers.append(
            nn.ReLU()
        )

        self.feature_layers = nn.Sequential(*feature_layers)

        self.value_layers = nn.Sequential(
            nn.Linear(hidden_sizes[-2], hidden_sizes[-1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[-1], 1)
        )

        self.advantage_layers = nn.Sequential(
            nn.Linear(hidden_sizes[-2], hidden_sizes[-1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[-1], output_dims)
        )

    def forward(self, x):
        features = self.feature_layers(x)
        
        value = self.value_layers(features)
        advantage = self.advantage_layers(features)

        q_value = value + advantage - torch.mean(advantage, dim=-1, keepdim=True)
        
        return q_value