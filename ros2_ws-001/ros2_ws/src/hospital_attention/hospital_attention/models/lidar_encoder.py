import torch
import torch.nn as nn


class LidarEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.ReLU(),

            nn.Linear(256, 512),
            nn.ReLU()
        )

        self.feature_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 128)
        )

    def forward(self, x):

        # x = (B,N,3)

        x = self.encoder(x)

        # x = (B,N,512)

        x = torch.max(x, dim=1)[0]

        # x = (B,512)

        x = self.feature_head(x)

        # x = (B,128)

        return x

