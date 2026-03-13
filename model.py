import torch
import torch.nn as nn

class PPGtoBPNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(1,16,7,padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16,32,5,padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32,64,3,padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.fc = nn.Sequential(
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,2)
        )

    def forward(self,x):
        x = self.conv(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x
