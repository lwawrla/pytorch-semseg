import torch.nn.functional as F
import torch, torchvision
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import *
from torchvision.models.resnet import Bottleneck

# model = {'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'}

class myresnet(torch.nn.Module):
    def __init__(self, num_classes):
        super(myresnet, self).__init__()
        self.num_classes = num_classes
        # load model
        self.model = torchvision.models.resnet101(pretrained=True)
        # torch.utils.model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        model.load.state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'))
        del self.model.avgpool
        del self.model.fc
        self.model.lastlayer = torch.nn.Conv2d

        # adapt with new layers
    def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.lastlayer(x)

            return x

        # input should fit output --> 32


a = ResNet(Bottleneck, [3, 4, 23, 4])