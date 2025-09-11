import torch.nn as nn


class LinearProjector(nn.Module):
    def __init__(self, in_hidden_size, out_hidden_size, bias=True):
        super().__init__()
        self.projector = nn.Linear(in_hidden_size, out_hidden_size, bias=bias)

    def forward(self, feature):
        return self.projector(feature)
