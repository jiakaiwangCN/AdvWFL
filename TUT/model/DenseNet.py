import torch

from model.MixtureNorm import MixtureNorm1d


class DenseLayer(torch.nn.Module):
    def __init__(self, in_channels, middle_channels=128, growth=32, device='cpu'):
        super(DenseLayer, self).__init__()
        self.device = device
        self.norm1 = MixtureNorm1d(in_channels)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.conv1 = torch.nn.Conv1d(in_channels, middle_channels, 1)
        self.norm2 = MixtureNorm1d(middle_channels)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv1d(middle_channels, in_channels + growth, 3, padding=1)

    def forward(self, x, data_type='normal'):
        if next(self.norm1.parameters()).device != self.device:
            self.norm1 = self.norm1.to(self.device)
        if next(self.norm2.parameters()).device != self.device:
            self.norm2 = self.norm2.to(self.device)
        if next(self.conv1.parameters()).device != self.device:
            self.conv1 = self.conv1.to(self.device)
        if next(self.conv2.parameters()).device != self.device:
            self.conv2 = self.conv2.to(self.device)
        x = self.norm1(x, data_type)
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.norm2(x, data_type)
        x = self.relu2(x)
        x = self.conv2(x)
        return x


class DenseBlock(torch.nn.Module):
    def __init__(self, layer_num, growth_rate, in_channels, middele_channels=128, device='cpu'):
        super(DenseBlock, self).__init__()
        self.device = device
        self.layers = []
        for i in range(layer_num):
            layer = DenseLayer(in_channels + i * growth_rate, middele_channels, growth_rate, device)
            self.layers.append(layer)

    def forward(self, x, data_type='normal'):
        for layer in self.layers:
            x = layer(x, data_type)
        return x


class Transition(torch.nn.Module):
    def __init__(self, channels, device='cpu'):
        super(Transition, self).__init__()
        self.device = device
        self.norm = MixtureNorm1d(channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv = torch.nn.Conv1d(channels, channels // 2, 3, padding=1)
        self.pool = torch.nn.AvgPool1d(2)

    def forward(self, x, data_type='normal'):
        x = self.norm(x, data_type)
        x = self.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x


class DenseNet(torch.nn.Module):
    def __init__(self, layer_num=(6, 12, 24, 16), growth_rate=32, init_features=64, in_channels=1, middele_channels=128,
                 classes=5, device='cpu'):
        super(DenseNet, self).__init__()
        self.feature_channel_num = init_features
        self.conv = torch.nn.Conv1d(in_channels, self.feature_channel_num, 7, 2, 3)
        self.norm = MixtureNorm1d(self.feature_channel_num)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool1d(3, 2, 1)

        self.DenseBlock1 = DenseBlock(layer_num[0], growth_rate, self.feature_channel_num, middele_channels, device)
        self.feature_channel_num = self.feature_channel_num + layer_num[0] * growth_rate
        self.Transition1 = Transition(self.feature_channel_num)

        self.DenseBlock2 = DenseBlock(layer_num[1], growth_rate, self.feature_channel_num // 2, middele_channels, device)
        self.feature_channel_num = self.feature_channel_num // 2 + layer_num[1] * growth_rate
        self.Transition2 = Transition(self.feature_channel_num)

        self.DenseBlock3 = DenseBlock(layer_num[2], growth_rate, self.feature_channel_num // 2, middele_channels, device)
        self.feature_channel_num = self.feature_channel_num // 2 + layer_num[2] * growth_rate
        self.Transition3 = Transition(self.feature_channel_num)

        self.DenseBlock4 = DenseBlock(layer_num[3], growth_rate, self.feature_channel_num // 2, middele_channels, device)
        self.feature_channel_num = self.feature_channel_num // 2 + layer_num[3] * growth_rate

        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)

        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(self.feature_channel_num, self.feature_channel_num // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(self.feature_channel_num // 2, classes),

        )

    def forward(self, x, data_type='normal'):
        x = self.conv(x)
        x = self.norm(x, data_type)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.DenseBlock1(x, data_type)
        x = self.Transition1(x, data_type)

        x = self.DenseBlock2(x, data_type)
        x = self.Transition2(x, data_type)

        x = self.DenseBlock3(x, data_type)
        x = self.Transition3(x, data_type)

        x = self.DenseBlock4(x, data_type)
        x = self.avgpool(x)
        x = x.view(-1, self.feature_channel_num)
        x = self.classifer(x)

        return x
