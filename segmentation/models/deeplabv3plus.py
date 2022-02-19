import torch
import torch.nn as nn
from torchvision.models.resnet import resnet101
import torch.nn.functional as F


class aspp_model(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding, dilation):
        super(aspp_model, self).__init__()
        self.atrous_conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.batchnorm = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x


class ASPP(nn.Module):
    def  __init__(self, in_channel = 2048):
        super(ASPP, self).__init__()
        dilation = [1, 6, 12, 18]
        self.aspp1 = aspp_model(in_channel, 256, 1, padding=0, dilation=dilation[0])
        self.aspp2 = aspp_model(in_channel, 256, 3, padding=dilation[1], dilation=dilation[1])
        self.aspp3 = aspp_model(in_channel, 256, 3, padding=dilation[2], dilation=dilation[2])
        self.aspp4 = aspp_model(in_channel, 256, 3, padding=dilation[3], dilation=dilation[3])
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channel, 256, 1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv = nn.Conv2d(1280, 256, 1, bias=False)
        self.batchnorm = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x


class decoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(decoder, self).__init__()
        self.conv = nn.Conv2d(in_channel, 48, 1, bias=False)
        self.bn = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(
            nn.Conv2d(304, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, out_channel, 1, stride=1)
        )

    def forward(self, x, feature):
        feature = self.conv(feature)
        feature = self.bn(feature)
        feature = self.relu(feature)
        x = F.interpolate(x, size=feature.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, feature), dim=1)
        x = self.last_conv(x)
        return x

# class decoder2(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super(decoder2, self).__init__()
#         self.conv = nn.Conv2d(in_channel, 48, 1, bias=False)
#         self.bn = nn.BatchNorm2d(48)
#         self.relu = nn.ReLU()
#         self.last_conv = nn.Sequential(
#             nn.Conv2d(50, 32, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.Conv2d(32, 32, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.Conv2d(32, out_channel, 1, stride=1)
#         )
#
#     def forward(self, x, feature):
#         feature = self.conv(feature)
#         feature = self.bn(feature)
#         feature = self.relu(feature)
#         x = F.interpolate(x, size=feature.size()[2:], mode='bilinear', align_corners=True)
#         x = torch.cat((x, feature), dim=1)
#         x = self.last_conv(x)
#         return x

class DeepLabV3Plus(nn.Module):
    def __init__(self, out_channel, pretrain=False):
        super(DeepLabV3Plus, self).__init__()
        self.backbone = resnet101(pretrained=pretrain)
        self.aspp = ASPP()
        self.decode = decoder(256, out_channel)
        # self.decode2 = decoder2(64, out_channel)

    def forward(self, x):
        sp = x.size()
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        # y = x
        x = self.backbone.layer1(x)
        feature = x
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.aspp(x)
        x = self.decode(x, feature)
        # x = self.decode2(x, y)
        x = F.interpolate(x, size=sp[2:], mode='bilinear', align_corners=True)
        return x


if __name__ == '__main__':
    input = torch.rand([3, 3, 512, 512], requires_grad=True)
    model = DeepLabV3Plus(out_channel=2)
    output = model(input)
    print(output.shape)