import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dropblock import DropBlock
from models.FiLM import FiLM_Layer
from models.dual_bn import DualBN2d

# This ResNet network was designed following the practice of the following
# papers:
#   Finding Task-Relevant Features for Few-Shot Learning by Category Traversal


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 drop_rate=0.0, drop_block=False, block_size=1,
                 film_indim=1, film_alpha=1, film_act=F.leaky_relu, dual_BN=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = DualBN2d(planes) if dual_BN else nn.BatchNorm2d(planes)
        self.film1 = FiLM_Layer(planes, in_channels=film_indim, alpha=film_alpha, activation=film_act)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = DualBN2d(planes) if dual_BN else nn.BatchNorm2d(planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.film2 = FiLM_Layer(planes, in_channels=film_indim, alpha=film_alpha, activation=film_act)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(planes * 4)
        self.bn3 = DualBN2d(planes * 4) if dual_BN else nn.BatchNorm2d(planes * 4)
        self.film3 = FiLM_Layer(planes * 4, in_channels=film_indim, alpha=film_alpha, activation=film_act)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if self.downsample:
            self.conv_ds = nn.Conv2d(inplanes, planes * self.expansion,
                                     kernel_size=1, stride=1, bias=False)
            if dual_BN:
                self.bn_ds = DualBN2d(planes * self.expansion)
            else:
                self.bn_ds = nn.BatchNorm2d(planes * self.expansion)
            self.film_ds = FiLM_Layer(
                planes * self.expansion, in_channels=film_indim,
                alpha=film_alpha, activation=film_act)


    def forward(self, x, task_embedding):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out, task_embedding) if self.dual_BN else self.bn1(out)
        out = self.film1(out, task_embedding)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, task_embedding) if self.dual_BN else self.bn2(out)
        out = self.film2(out, task_embedding)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out, task_embedding) if self.dual_BN else self.bn3(out)
        out = self.film3(out, task_embedding)

        # if self.downsample is not None:
        #     residual = self.downsample(x)
        if self.downsample:
            residual = self.conv_ds(x)
            residual = self.bn_ds(residual, task_embedding) if self.dual_BN else self.bn_ds(residual)
            residual = self.film_ds(residual, task_embedding)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, in_c, drop_rate=0.0, dropblock_size=5,
                 film_indim=1, film_alpha=1, film_act=F.leaky_relu, dual_BN=False):
        super(ResNet, self).__init__()

        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_c, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        self.bn1 = DualBN2d(64) if dual_BN else nn.BatchNorm2d(64)
        self.film1 = FiLM_Layer(64, in_channels=film_indim, alpha=film_alpha, activation=film_act)

        self.relu = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0],
            film_indim=film_indim, film_alpha=film_alpha, film_act=film_act, dual_BN=dual_BN)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
            film_indim=film_indim, film_alpha=film_alpha, film_act=film_act, dual_BN=dual_BN)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
            film_indim=film_indim, film_alpha=film_alpha, film_act=film_act, dual_BN=dual_BN)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.dual_BN = dual_BN

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu')

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        # downsample = None
        downsample = stride != 1 or self.inplanes != planes * block.expansion
        # if stride != 1 or self.inplanes != planes * block.expansion:
        #     downsample = nn.Sequential(
        #         nn.Conv2d(self.inplanes, planes * block.expansion,
        #                   kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(planes * block.expansion),
        #     )

        # layers = []
        layers = nn.ModuleList()
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        # return nn.Sequential(*layers)
        return layers

    def forward(self, x, task_embedding):
        x = self.conv1(x)
        x = self.bn1(x, task_embedding) if self.dual_BN else self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for m in self.layer1:
            x = m(x, task_embedding)
        for m in self.layer2:
            x = m(x, task_embedding)
        for m in self.layer3:
            x = m(x, task_embedding)

        return x


def feat_extract(pretrained=False, **kwargs):
    """Constructs a ResNet-Mini-Imagenet model"""
    model_urls = {
        'resnet18':     'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34':     'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet52':     'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101':    'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152':    'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    }
    logger = kwargs['opts'].logger
    # resnet"x", x = 1 + sum(layers)x3
    if kwargs['structure'] == 'resnet40':
        model = ResNet(Bottleneck, [3, 4, 6], kwargs['in_c'], **kwargs)
    elif kwargs['structure'] == 'resnet19':
        model = ResNet(Bottleneck, [2, 2, 2], kwargs['in_c'], **kwargs)
    elif kwargs['structure'] == 'resnet52':
        model = ResNet(Bottleneck, [4, 8, 5], kwargs['in_c'], **kwargs)
    elif kwargs['structure'] == 'resnet34':
        model = ResNet(Bottleneck, [3, 4, 4], kwargs['in_c'], **kwargs)
    elif kwargs['structure'] == 'shallow':
        model = CNNEncoder(kwargs['in_c'], **kwargs)
    else:
        raise NameError('structure not known {} ...'.format(kwargs['structure']))
    if pretrained:
        logger('Using pre-trained model from pytorch official webiste, {:s}'.format(kwargs['structure']))
        model.load_state_dict(model_zoo.load_url(model_urls[kwargs['structure']]), strict=False)
    return model


