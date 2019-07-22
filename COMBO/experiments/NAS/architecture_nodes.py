import torch.nn as nn


class NodeConv(nn.Module):
    def __init__(self, n_channels):
        super(NodeConv, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=n_channels)
        self.relu = nn.ReLU()

    def init_weights(self):
        nn.init.normal_(self.bn.weight, 1.0, 0.02)
        nn.init.constant_(self.bn.bias, 0)
        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class NodeConv5by5(NodeConv):
    def __init__(self, n_channels):
        super(NodeConv5by5, self).__init__(n_channels)
        self.conv = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=5, padding=2, bias=True)


class NodeConv3by3(NodeConv):
    def __init__(self, n_channels):
        super(NodeConv3by3, self).__init__(n_channels)
        self.conv = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, padding=1, bias=True)


class NodeConv1by1(NodeConv):
    def __init__(self, n_channels):
        super(NodeConv1by1, self).__init__(n_channels)
        self.conv = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=1, padding=0, bias=True)


class NodeMaxpool3by3(nn.Module):
    def __init__(self):
        super(NodeMaxpool3by3, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def init_weights(self):
        pass

    def forward(self, x):
        return self.maxpool(x)


# Id can be considered as 1 by 1 max pool
class NodeId(nn.Module):
    def __init__(self):
        super(NodeId, self).__init__()

    def init_weights(self):
        pass

    def forward(self, x):
        return x