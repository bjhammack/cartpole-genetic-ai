import torch.nn as nn


class TwoLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(4,128, bias=True),
            nn.ReLU(),
            nn.Linear(128,2, bias=True),
            nn.Softmax(dim=1)
            )
        
    def forward(self, inputs):
        x = self.fc(inputs)
        return x