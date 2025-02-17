from model.AlexNet import AlexNet
from model.DenseNet import DenseNet
from model.ResNet import ResNet
from model.VGG import VGG


class Model:
    def __init__(self, type='ResNet', in_channels=2, classes=5, device='cpu', data_types=None):
        self.device = device
        assert type in ['VGG', 'AlexNet', 'ResNet']
        self.model = None
        if type == 'VGG':
            self.model = VGG(in_channels=in_channels, classes=classes, data_types=data_types)
        elif type == 'AlexNet':
            self.model = AlexNet(in_channels=in_channels, classes=classes, data_types=data_types)
        elif type == 'ResNet':
            self.model = ResNet(in_channels=in_channels, classes=classes, data_types=data_types)


