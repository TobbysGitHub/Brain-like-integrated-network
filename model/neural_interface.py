import torch.nn as nn


class NeuralInterface(nn.Module):
    def __init__(self, dim_inputs, dim_hidden, dim_outputs):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.dim_hidden = dim_hidden
        self.dim_outputs = dim_outputs

        self.model = nn.Sequential(
            nn.Linear(in_features=dim_inputs, out_features=dim_hidden),
            nn.ReLU(),
            nn.Linear(in_features=dim_hidden, out_features=dim_outputs)
        )

        self.apply(self.init)

    def init(self):
        for p in self.parameters():
            nn.init.xavier_normal_(p)

    def forward(self, x):
        return self.model(x)
