import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import models.layers as nl
import pdb

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19', 'custom_vgg', 'custom_vgg_cifar100'
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class View(nn.Module):
    """Changes view using a nn.Module."""

    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)

class VGG(nn.Module):
    def __init__(self, features, dataset_history, dataset2num_classes, network_width_multiplier=1.0, shared_layer_info={}, init_weights=True, progressive_init=False):
        super(VGG, self).__init__()
        self.features = features
        self.network_width_multiplier = network_width_multiplier
        self.shared_layer_info = shared_layer_info
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.datasets, self.classifiers = dataset_history, nn.ModuleList()
        self.dataset2num_classes = dataset2num_classes

        if self.datasets:
            self._reconstruct_classifiers()

        if init_weights:
            self._initialize_weights()

        if progressive_init:
            self._initialize_weights_2()

    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nl.SharableConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nl.SharableLinear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _initialize_weights_2(self):
        for m in self.modules():
            if isinstance(m, nl.SharableConv2d):
                nn.init.normal_(m.weight, 0, 0.01)

    def _reconstruct_classifiers(self):
        for dataset, num_classes in self.dataset2num_classes.items():
            self.classifiers.append(nn.Linear(int(self.shared_layer_info[dataset]['network_width_multiplier'] * 4096), num_classes))

    def add_dataset(self, dataset, num_classes):
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            self.dataset2num_classes[dataset] = num_classes
            self.classifiers.append(nn.Linear(int(4096*self.network_width_multiplier), num_classes))
            nn.init.normal_(self.classifiers[self.datasets.index(dataset)].weight, 0, 0.01)
            nn.init.constant_(self.classifiers[self.datasets.index(dataset)].bias, 0)

    def set_dataset(self, dataset):
        """Change the active classifier."""
        assert dataset in self.datasets
        self.classifier = self.classifiers[self.datasets.index(dataset)]

def make_layers_cifar100(cfg, network_width_multiplier, batch_norm=False, groups=1):
    layers = []
    in_channels = 3

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if in_channels == 3:
                conv2d = nl.SharableConv2d(in_channels, int(v * network_width_multiplier), kernel_size=3, padding=1, bias=False)
            else:
                conv2d = nl.SharableConv2d(in_channels, int(v * network_width_multiplier), kernel_size=3, padding=1, bias=False, groups=groups)

            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(int(v * network_width_multiplier)), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = int(v * network_width_multiplier)

    layers += [
        View(-1, int(512*network_width_multiplier)),
        nl.SharableLinear(int(512*network_width_multiplier), int(4096*network_width_multiplier)),
        nn.ReLU(True),
        nl.SharableLinear(int(4096*network_width_multiplier), int(4096*network_width_multiplier)),
        nn.ReLU(True),
    ]

    return nn.Sequential(*layers)

def make_layers(cfg, network_width_multiplier, batch_norm=False, groups=1):
    layers = []
    in_channels = 3

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if in_channels == 3:
                conv2d = nl.SharableConv2d(in_channels, int(v * network_width_multiplier), kernel_size=3, padding=1, bias=False)
            else:
                conv2d = nl.SharableConv2d(in_channels, int(v * network_width_multiplier), kernel_size=3, padding=1, bias=False, groups=groups)

            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(int(v * network_width_multiplier)), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = int(v * network_width_multiplier)

    layers += [
        View(-1, int(512*network_width_multiplier)*7*7),
        nl.SharableLinear(int(512*network_width_multiplier)*7*7, int(4096*network_width_multiplier)),
        nn.ReLU(True),
        # We need Dropout() for 224x224
        nn.Dropout(),
        nl.SharableLinear(int(4096*network_width_multiplier), int(4096*network_width_multiplier)),
        nn.ReLU(True),
        nn.Dropout()
    ]

    return nn.Sequential(*layers)

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(pretrained=False, dataset_history=[], dataset2num_classes={}, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A']), dataset_history, dataset2num_classes, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model


def vgg11_bn(pretrained=False, dataset_history=[], dataset2num_classes={}, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], batch_norm=True), dataset_history, dataset2num_classes, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model


def vgg13(pretrained=False, dataset_history=[], dataset2num_classes={}, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B']), dataset_history, dataset2num_classes, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13']))
    return model


def vgg13_bn(pretrained=False, dataset_history=[], dataset2num_classes={}, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'], batch_norm=True), dataset_history, dataset2num_classes, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']))
    return model


def vgg16(pretrained=False, dataset_history=[], dataset2num_classes={}, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), dataset_history, dataset2num_classes, **kwargs)
    
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model


def vgg16_bn(pretrained=False, dataset_history=[], dataset2num_classes={}, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True), dataset_history, dataset2num_classes, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model


def vgg19(pretrained=False, dataset_history=[], dataset2num_classes={}, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E']), dataset_history, dataset2num_classes, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model


def vgg19_bn(pretrained=False, dataset_history=[], dataset2num_classes={}, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], batch_norm=True), dataset_history, dataset2num_classes, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model

def custom_vgg_cifar100(custom_cfg, dataset_history=[], dataset2num_classes={}, network_width_multiplier=1.0, groups=1, shared_layer_info={}, **kwargs):
    return VGG(make_layers_cifar100(custom_cfg, network_width_multiplier, batch_norm=True, groups=groups), dataset_history, 
        dataset2num_classes, network_width_multiplier, shared_layer_info, **kwargs)

def custom_vgg(custom_cfg, dataset_history=[], dataset2num_classes={}, network_width_multiplier=1.0, groups=1, shared_layer_info={}, **kwargs):
    return VGG(make_layers(custom_cfg, network_width_multiplier, batch_norm=True, groups=groups), dataset_history, 
        dataset2num_classes, network_width_multiplier, shared_layer_info, **kwargs)