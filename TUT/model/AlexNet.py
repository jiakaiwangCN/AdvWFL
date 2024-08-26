import torch

from model.MixtureNorm import MixtureNorm1d


class AlexNet(torch.nn.Module):
    def __init__(self, in_channels, classes, data_types):
        super(AlexNet, self).__init__()

        self.conv1 = torch.nn.Conv1d(in_channels, 64, kernel_size=11, stride=4, padding=2)
        self.norm1 = MixtureNorm1d(64, data_types)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.pool1 = torch.nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv2 = torch.nn.Conv1d(64, 192, kernel_size=5, padding=2)
        self.norm2 = MixtureNorm1d(192, data_types)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.pool2 = torch.nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv3 = torch.nn.Conv1d(192, 384, kernel_size=3, padding=1)
        self.norm3 = MixtureNorm1d(384, data_types)
        self.relu3 = torch.nn.ReLU(inplace=True)
        self.conv4 = torch.nn.Conv1d(384, 256, kernel_size=3, padding=1)
        self.relu4 = torch.nn.ReLU(inplace=True)
        self.norm4 = MixtureNorm1d(256, data_types)
        self.conv5 = torch.nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.norm5 = MixtureNorm1d(256, data_types)
        self.relu5 = torch.nn.ReLU(inplace=True)
        self.pool3 = torch.nn.MaxPool1d(kernel_size=3, stride=2)
        self.pool4 = torch.nn.AdaptiveAvgPool1d(6)

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1536, 1024),
            torch.nn.ReLU(inplace=True),

            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(inplace=True),
        )

        self.outputs1 = torch.nn.Linear(1024, 1)
        self.outputs2 = torch.nn.Linear(1024, 1)

    def forward(self, x, data_type='normal'):
        x = self.conv1(x)
        x = self.norm1(x, data_type)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.norm2(x, data_type)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.norm3(x, data_type)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.norm4(x, data_type)
        x = self.conv5(x)
        x = self.norm5(x, data_type)
        x = self.relu5(x)
        x = self.pool3(x)
        x = self.pool4(x)
        x = x.view(-1, 1536)
        x = self.classifier(x)
        x1 = self.outputs1(x)
        x2 = self.outputs2(x)
        return x1, x2
