# NET DESIGNS
import torch.nn as nn


class KDDClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(KDDClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes),
            nn.LogSoftmax()
        )

    def forward(self, x):
        out = self.model(x)
        return out


class KDDRegressor(nn.Module):
    def __init__(self):
        super(KDDRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
        )

    def forward(self, x):
        out = self.model(x).squeeze()
        return out


class VSRegressor(nn.Module):
    def __init__(self):
        super(VSRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(20 * 8, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
        )

    def forward(self, x):
        out = self.model(x)
        return out
