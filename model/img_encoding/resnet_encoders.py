import torch.nn as nn
import torchvision.models as models


class ResNetEncoder(nn.Module):
    def __init__(self, out_size, model_name, pretrained,
                 doubled_res=True, freeze=True):
        super(ResNetEncoder, self).__init__()

        assert model_name in ['resnet34', 'resnet50', 'resnet18']

        if model_name == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
        elif model_name == 'resnet34':
            model = models.resnet34(pretrained=pretrained)
        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=pretrained)

        layers = list(model.children())[:-2]
        self.model = nn.Sequential(*layers)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        # resnet_out_ch = 512 if model_name in ['resnet18', 'resnet34'] else 2048
        # print(layers)
        # print(list(layers[-1][-1].children()))
        # print(layers[-1][-1].conv3)
        # if 'conv3' in layers[-1][-1]:
        #     print(layers[-1][-1])
        #     print(layers[-1][-1].conv3)
        # else:
        #     print(layers[-1][-1].conv2)

        conv_list = []
        if model_name not in ['resnet18', 'resnet34']:
            conv_list.append(nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0))
            conv_list.append(nn.BatchNorm2d(512))

        conv_list.append(nn.ReLU(inplace=True))
        if doubled_res:
            conv_list.append(nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1))
        else:
            conv_list.append(nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0))
        conv_list.append(nn.BatchNorm2d(256))
        conv_list.append(nn.ReLU(inplace=True))
        conv_list.append(nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0))
        conv_list.append(nn.BatchNorm2d(64))
        conv_list.append(nn.ReLU(inplace=True))

        self.model_head = nn.Sequential(*conv_list)
        # print(self.model)
        # print(self.model_head)
        self.flattened_size = 64 * 7 * 7
        self.linear_layer = nn.Linear(self.flattened_size, out_size)
        self.out_size = out_size

    def forward(self, inp):
        out = self.model(inp)
        out = self.model_head(out)
        # print(out.shape)
        out = out.view(-1, self.flattened_size)
        out = self.linear_layer(out)
        return out


class ResNetEncoderNoLin(nn.Module):
    def __init__(self, out_channels, model_name, pretrained,
                 doubled_res=True, freeze=False):
        super(ResNetEncoderNoLin, self).__init__()

        assert model_name in ['resnet34', 'resnet50', 'resnet18']

        if model_name == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
        elif model_name == 'resnet34':
            model = models.resnet34(pretrained=pretrained)
        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=pretrained)

        layers = list(model.children())[:-2]
        self.model = nn.Sequential(*layers)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        # resnet_out_ch = 512 if model_name in ['resnet18', 'resnet34'] else 2048
        # print(layers)
        # print(list(layers[-1][-1].children()))
        # print(layers[-1][-1].conv3)
        # if 'conv3' in layers[-1][-1]:
        #     print(layers[-1][-1])
        #     print(layers[-1][-1].conv3)
        # else:
        #     print(layers[-1][-1].conv2)

        conv_list = []
        if model_name not in ['resnet18', 'resnet34']:
            conv_list.append(nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0))
            conv_list.append(nn.BatchNorm2d(512))

        conv_list.append(nn.ReLU(inplace=True))
        conv_list.append(nn.Conv2d(512, out_channels, kernel_size=1, stride=1, padding=0))

        self.model_head = nn.Sequential(*conv_list)
        self.out_channels = out_channels
        # print(self.model)
        # print(self.model_head)

    def forward(self, inp):
        out = self.model(inp)
        out = self.model_head(out)
        return out
