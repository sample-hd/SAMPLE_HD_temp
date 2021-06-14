import torch.nn as nn


class ScaledSigmoid(nn.Module):
    def __init__(self, val_range):
        super(ScaledSigmoid, self).__init__()
        range_type = isinstance(val_range, tuple) or isinstance(val_range, list)
        range_len = len(val_range) == 2
        assert range_type and range_len, "Provide two values for the range"
        self.scale = val_range[1] - val_range[0]
        self.shift = val_range[0]
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp):
        return self.sigmoid(inp) * self.scale + self.shift


class TestSigmoid(nn.Module):
    def __init__(self):
        super(TestSigmoid, self).__init__()
        # range_type = isinstance(val_range, tuple) or isinstance(val_range, list)
        # range_len = len(val_range) == 2
        # assert range_type and range_len, "Provide two values for the range"
        self.scale = 200
        self.shift = -100
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp):
        return (self.sigmoid(inp) * self.scale + self.shift) / 100
        # return inp / 100
