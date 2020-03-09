import torch
import torch.nn as nn


class ImageGenNet(nn.Module):
    def __init__(self, dim_inputs):
        super().__init__()
        self.dim_inputs = dim_inputs

        self.model = nn.Sequential(
            nn.Linear(in_features=dim_inputs,
                      out_features=256),
            self.Reshape(-1, 256, 1, 1),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=6, stride=3),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=6, stride=2),
            nn.Tanh(),
            self.Reshape(-1, 96 * 96)
        )

        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.view(-1, self.dim_inputs)
        x = self.model(x)
        return x

    class Reshape(nn.Module):
        def __init__(self, *args):
            super().__init__()
            self.shape = args

        def forward(self, x):
            return x.view(self.shape)
