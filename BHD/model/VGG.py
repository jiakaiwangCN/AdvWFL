import torch

from model.MixtureNorm import MixtureNorm1d


class VGGBlock1(torch.nn.Module):
    def __init__(self, in_channels, out_channels, data_types):
        super(VGGBlock1, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = MixtureNorm1d(out_channels, data_types)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = MixtureNorm1d(out_channels, data_types)
        self.relu2 = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool1d(2)

    def forward(self, x, data_type='normal'):
        x = self.conv1(x)
        x = self.norm1(x, data_type)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.norm2(x, data_type)
        x = self.relu2(x)
        x = self.pool(x)
        return x


class VGGBlock2(torch.nn.Module):
    def __init__(self, in_channels, out_channels, data_types):
        super(VGGBlock2, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = MixtureNorm1d(out_channels, data_types)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = MixtureNorm1d(out_channels, data_types)
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm3 = MixtureNorm1d(out_channels, data_types)
        self.relu3 = torch.nn.ReLU()
        self.conv4 = torch.nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm4 = MixtureNorm1d(out_channels, data_types)
        self.relu4 = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool1d(2)

    def forward(self, x, data_type='normal'):
        x = self.conv1(x)
        x = self.norm1(x, data_type)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.norm2(x, data_type)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x, data_type)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.norm4(x, data_type)
        x = self.relu4(x)
        x = self.pool(x)
        return x


class VGG(torch.nn.Module):
    def __init__(self, in_channels=2, classes=5, data_types=None):
        super(VGG, self).__init__()
        self.block1 = VGGBlock1(in_channels, 64, data_types)
        self.block2 = VGGBlock1(64, 128, data_types)
        self.block3 = VGGBlock2(128, 256, data_types)
        self.block4 = VGGBlock2(256, 512, data_types)
        self.block5 = VGGBlock2(512, 512, data_types)
        self.pool = torch.nn.AdaptiveAvgPool1d(7)
        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(3584, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
        )
        self.output1 = torch.nn.Linear(512, 1)
        self.output2 = torch.nn.Linear(512, 1)

    def forward(self, x, data_type='normal'):
        x = self.block1(x, data_type)
        x = self.block2(x, data_type)
        x = self.block3(x, data_type)
        x = self.block4(x, data_type)
        x = self.block5(x, data_type)
        x = self.pool(x)
        x = x.view(-1, 3584)
        x = self.classifer(x)
        x1 = self.output1(x)
        x2 = self.output2(x)
        return x1, x2

