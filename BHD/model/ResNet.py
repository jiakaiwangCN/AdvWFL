import torch

from model.MixtureNorm import MixtureNorm1d


class ResidualBlock(torch.nn.Module):
    def __init__(self, In_channel, Med_channel, Out_channel, downsample=False, data_types=None):
        super(ResidualBlock, self).__init__()
        self.stride = 1
        if downsample:
            self.stride = 2

        self.conv1 = torch.nn.Conv1d(In_channel, Med_channel, 1, self.stride)
        self.norm1 = MixtureNorm1d(Med_channel, data_types)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(Med_channel, Med_channel, 3, padding=1)
        self.norm2 = MixtureNorm1d(Med_channel, data_types)
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv1d(Med_channel, Out_channel, 1)
        self.norm3 = MixtureNorm1d(Out_channel, data_types)
        self.relu3 = torch.nn.ReLU()

        if In_channel != Out_channel:
            self.res_layer = torch.nn.Conv1d(In_channel, Out_channel, 1, self.stride)
        else:
            self.res_layer = None

    def forward(self, x, data_type='normal'):
        if self.res_layer is not None:
            residual = self.res_layer(x)
        else:
            residual = x
        x = self.conv1(x)
        x = self.norm1(x, data_type)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.norm2(x, data_type)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x, data_type)
        x = self.relu3(x)

        return x + residual


class ResNet(torch.nn.Module):
    def __init__(self, in_channels=2, classes=5, data_types=None):
        super(ResNet, self).__init__()
        self.maxpool = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            torch.nn.MaxPool1d(3, 2, 1),
        )
        self.rb1 = ResidualBlock(64, 64, 256, False, data_types)
        self.rb2 = ResidualBlock(256, 64, 256, False, data_types)
        self.rb3 = ResidualBlock(256, 64, 256, False, data_types)
        self.rb4 = ResidualBlock(256, 128, 512, True, data_types)
        self.rb5 = ResidualBlock(512, 128, 512, False, data_types)
        self.rb6 = ResidualBlock(512, 128, 512, False, data_types)
        self.rb7 = ResidualBlock(512, 128, 512, False, data_types)
        self.rb8 = ResidualBlock(512, 256, 1024, True, data_types)
        self.rb9 = ResidualBlock(1024, 256, 1024, False, data_types)
        self.rb10 = ResidualBlock(1024, 256, 1024, False, data_types)
        self.rb11 = ResidualBlock(1024, 256, 1024, False, data_types)
        self.rb12 = ResidualBlock(1024, 256, 1024, False, data_types)
        self.rb13 = ResidualBlock(1024, 256, 1024, False, data_types)
        self.rb14 = ResidualBlock(1024, 512, 2048, True, data_types)
        self.rb15 = ResidualBlock(2048, 512, 2048, False, data_types)
        self.rb16 = ResidualBlock(2048, 512, 2048, False, data_types)
        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)
        self.output1 = torch.nn.Linear(2048, 1)
        self.output2 = torch.nn.Linear(2048, 1)

    def forward(self, x, data_type='normal', ret_pool_value=False):
        x = self.maxpool(x)
        x = self.rb1(x, data_type)
        x = self.rb2(x, data_type)
        x = self.rb3(x, data_type)
        x = self.rb4(x, data_type)
        x = self.rb5(x, data_type)
        x = self.rb6(x, data_type)
        x = self.rb7(x, data_type)
        x = self.rb8(x, data_type)
        x = self.rb9(x, data_type)
        x = self.rb10(x, data_type)
        x = self.rb11(x, data_type)
        x = self.rb12(x, data_type)
        x = self.rb13(x, data_type)
        x = self.rb14(x, data_type)
        x = self.rb15(x, data_type)
        x = self.rb16(x, data_type)
        x = self.avgpool(x)
        x = x.view(-1, 2048)
        temp = x
        x1 = self.output1(x)
        x2 = self.output2(x)
        if ret_pool_value:
            return x1, x2, temp
        return x1, x2
