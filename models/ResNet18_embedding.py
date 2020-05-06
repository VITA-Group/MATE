import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo
from models.dropblock import DropBlock

# This ResNet network was designed following the practice of the following
# papers:
#   Finding Task-Relevant Features for Few-Shot Learning by Category Traversal


eps = 1e-10


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 drop_rate=0.0, drop_block=False, block_size=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        if self.drop_rate > 0:
            if self.drop_block:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20*2000) * self.num_batches_tracked, 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size**2 * feat_size**2 / (feat_size - self.block_size + 1)**2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training,
                                inplace=True)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, in_c,
                 drop_rate=0.0, dropblock_size=5, **kwargs):
        super(ResNet, self).__init__()

        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_c, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(
            block, 64, layers[0], drop_rate=drop_rate)
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2,
            drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2,
            drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1,
                    drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            drop_rate, drop_block, block_size))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None,
                                drop_rate, drop_block, block_size))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x.view(x.size(0), -1)


def feat_extract(pretrained=False, **kwargs):
    """Constructs a ResNet-Mini-Imagenet model"""
    model_urls = {
        'resnet18':  'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34':  'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet52':  'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
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
    else:
        raise NameError('structure not known {} ...'.format(kwargs['structure']))
    if pretrained:
        logger('Using pre-trained model from pytorch official webiste, {:s}'.format(kwargs['structure']))
        model.load_state_dict(model_zoo.load_url(model_urls[kwargs['structure']]), strict=False)
    return model


def resnet18(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [2, 2, 2], in_c=3, **kwargs)
    return model


if __name__ == '__main__':
    res = resnet18()
    x = torch.rand(1, 3, 84, 84)
    with torch.no_grad():
        output = res(x)
    print(output.shape)