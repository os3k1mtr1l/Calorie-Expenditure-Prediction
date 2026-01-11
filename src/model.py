import torch
import torch.nn as nn

class ModelSmall(nn.Module):
    def __init__(self):
        super(ModelSmall, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x)


class ModelInitial(nn.Module):
    
    def __init__(self) -> None:
        super(ModelInitial, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
class ModelInitialDropout(nn.Module):
    
    def __init__(self, p: float = 0.3) -> None:
        super(ModelInitialDropout, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 8),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(4, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
class ModelLarge(nn.Module):
    def __init__(self):
        super(ModelLarge, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)