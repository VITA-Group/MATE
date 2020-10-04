import torch.nn as nn
import torch.nn.functional as F

from models.FiLM import FiLM_Layer
from models.dropblock import DropBlock
from models.dual_bn import DualBN2d


# This ResNet network was designed following the practice of the
#   following papers:
# TADAM: Task dependent adaptive metric for improved few-shot learning
#   (Oreshkin et al., in NIPS 2018) and
# A Simple Neural Attentive Meta-Learner (Mishra et al., in ICLR 2018).

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlockFiLM(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 drop_rate=0.0, drop_block=False, block_size=1, final_relu=True,
                 film_indim=1, film_alpha=1, film_act=F.leaky_relu,
                 film_normalize=True, dual_BN=True):
        super(BasicBlockFiLM, self).__init__()

        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = DualBN2d(planes) if dual_BN else nn.BatchNorm2d(planes)
        self.film1 = FiLM_Layer(planes, in_channels=film_indim,
                                alpha=film_alpha,
                                activation=film_act, normalize=film_normalize)
        self.relu = nn.LeakyReLU(0.1)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = DualBN2d(planes) if dual_BN else nn.BatchNorm2d(planes)
        self.film2 = FiLM_Layer(planes, in_channels=film_indim,
                                alpha=film_alpha,
                                activation=film_act, normalize=film_normalize)

        self.conv3 = conv3x3(planes, planes)
        self.bn3 = DualBN2d(planes) if dual_BN else nn.BatchNorm2d(planes)
        self.film3 = FiLM_Layer(planes, in_channels=film_indim,
                                alpha=film_alpha,
                                activation=film_act, normalize=film_normalize)

        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        if self.downsample:
            self.conv_ds = nn.Conv2d(inplanes, planes * self.expansion,
                                     kernel_size=1, stride=1, bias=False)
            if dual_BN:
                self.bn_ds = DualBN2d(planes * self.expansion)
            else:
                self.bn_ds = nn.BatchNorm2d(planes * self.expansion)
            self.film_ds = FiLM_Layer(
                planes * self.expansion, in_channels=film_indim,
                alpha=film_alpha, activation=film_act, normalize=film_normalize)

        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.dual_BN = dual_BN
        self.DropBlock = DropBlock(block_size=self.block_size)
        self.final_relu = final_relu

    def forward(self, x, task_embedding, n_expand):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out, task_embedding) if self.dual_BN else self.bn1(out)
        out = self.film1(out, task_embedding, n_expand)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, task_embedding) if self.dual_BN else self.bn2(out)
        out = self.film2(out, task_embedding, n_expand)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out, task_embedding) if self.dual_BN else self.bn3(out)
        out = self.film3(out, task_embedding, n_expand)

        if self.downsample:
            residual = self.conv_ds(x)
            residual = self.bn_ds(residual, task_embedding) if self.dual_BN else self.bn_ds(residual)
            residual = self.film_ds(residual, task_embedding, n_expand)

        out += residual
        if self.final_relu:
            out = self.relu(out)
        out = self.maxpool(out)

        if self.drop_rate > 0:
            if self.drop_block:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20 * 2000) * (
                    self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = ((1 - keep_rate) / self.block_size**2 *
                         feat_size**2 / (feat_size - self.block_size + 1)**2)
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training,
                                inplace=True)

        return out


class ResNet_FiLM(nn.Module):
    def __init__(self, block, keep_prob=1.0, avg_pool=False,
                 drop_rate=0.0, dropblock_size=5, final_relu=True,
                 film_indim=1, film_alpha=1, film_act=F.leaky_relu,
                 film_normalize=True, dual_BN=True):
        self.inplanes = 3
        super(ResNet_FiLM, self).__init__()

        self.layer1 = self._make_layer(
            block, 64, stride=2, drop_rate=drop_rate,
            film_indim=film_indim, film_alpha=film_alpha, film_act=film_act,
            film_normalize=film_normalize, dual_BN=dual_BN)
        self.layer2 = self._make_layer(
            block, 160, stride=2, drop_rate=drop_rate,
            film_indim=film_indim, film_alpha=film_alpha, film_act=film_act,
            film_normalize=film_normalize, dual_BN=dual_BN)
        self.layer3 = self._make_layer(
            block, 320, stride=2, drop_rate=drop_rate, drop_block=True,
            block_size=dropblock_size,
            film_indim=film_indim, film_alpha=film_alpha, film_act=film_act,
            film_normalize=film_normalize, dual_BN=dual_BN)
        self.layer4 = self._make_layer(
            block, 640, stride=2, drop_rate=drop_rate, drop_block=True,
            block_size=dropblock_size, final_relu=final_relu,
            film_indim=film_indim, film_alpha=film_alpha, film_act=film_act,
            film_normalize=film_normalize, dual_BN=dual_BN)
        if avg_pool:
            self.avgpool = nn.AvgPool2d(5, stride=1)

        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate
        self.final_relu = final_relu

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0,
                    drop_block=False, block_size=1, final_relu=True,
                    film_indim=1, film_alpha=1, film_act=F.leaky_relu,
                    film_normalize=True, dual_BN=True):
        downsample = stride != 1 or self.inplanes != planes * block.expansion
        layers = block(self.inplanes, planes, stride, downsample,
                       drop_rate, drop_block, block_size, final_relu,
                       film_indim, film_alpha, film_act, film_normalize,
                       dual_BN)
        self.inplanes = planes * block.expansion

        return layers

    def forward(self, x, task_embedding, n_expand):
        x = self.layer1(x, task_embedding, n_expand)
        x = self.layer2(x, task_embedding, n_expand)
        x = self.layer3(x, task_embedding, n_expand)
        x = self.layer4(x, task_embedding, n_expand)
        if self.keep_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


def resnet12_film(keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs a ResNet-12 model.
    """
    model = ResNet_FiLM(BasicBlockFiLM, keep_prob=keep_prob, avg_pool=avg_pool,
                        **kwargs)
    return model
