import torch.nn.functional as F
import torch
import torchvision
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import Bottleneck
from torchvision.models.resnet import ResNet



model_urls = {'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'}


class resnet(ResNet):
    def __init__(self, n_classes=19):
        super(resnet, self).__init__(Bottleneck, [3, 4, 23, 3])

        self.n_classes = n_classes

        # load model

        self.load_state_dict(model_zoo.load_url(model_urls['resnet101']))

        # delete layers

        del self.avgpool
        del self.fc

        # adapt with new layers

        #self.lastlayer = nn.Conv2d(2048, n_classes, kernel_size=3, padding=2)
        #self.deconv1= nn.ConvTranspose2d(2048, 1024, kernel_size = (8, 4), stride = (8, 4))
        #self.deconv2 = nn.ConvTranspose2d(1024, n_classes, kernel_size = (8, 4), stride = (8, 4))
        self.deconv = nn.ConvTranspose2d(2048, n_classes, kernel_size = (32,  32), stride = (32, 32))

    def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            #x = self.classifier(x)
            print(x.shape)
            #x = self.deconv1(x)
            print(x.shape)
            x = self.deconv(x)
            print(x.shape)
            return x


# a = ResNet(Bottleneck, [3, 4, 23, 3])
