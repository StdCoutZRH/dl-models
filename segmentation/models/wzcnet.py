import torch
import torch.nn as nn
import torch.nn.functional as F

num_classes = 2


class TextureMap(nn.Module):
    def __init__(self, scale=4):
        super().__init__()
        self.scale = scale
        self.conv1 = nn.Sequential(ConvBNReLU(64, 256, 3, 1, 1, has_relu=False), nn.LeakyReLU(inplace=True))
        self.conv2 = ConvBNReLU(256, 128, 3, 1, 1, has_bn=False, has_relu=False)
        self.se_conv1 = nn.Sequential(ConvBNReLU(128, 128, 1, 1, 0, has_relu=False), nn.LeakyReLU(inplace=True))
        self.se_conv2 = nn.Sequential(ConvBNReLU(128, 128, 1, 1, 0, has_relu=False))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        N1, C1, H1, W1 = x.shape
        if H1 // self.scale != 0 or W1 // self.scale != 0:
            x = F.adaptive_avg_pool2d(x, ((int(H1 / self.scale) * self.scale), int(W1 / self.scale) * self.scale))

        N, C, H, W = x.shape

        se = F.adaptive_avg_pool2d(x, (1, 1))
        se = self.se_conv2(self.se_conv1(se))
        se = F.sigmoid(se)
        x = x + se * x

        # scale = self.scale
        scale = 1

        x_ave = F.adaptive_avg_pool2d(x, (scale, scale))
        x_ave_up = F.adaptive_avg_pool2d(x_ave, (H, W))
        cos_sim = (F.normalize(x_ave_up, dim=1) * F.normalize(x, dim=1)).sum(1)

        return cos_sim, x


class TEMDistinct(nn.Module):
    def __init__(self):
        super().__init__()
        self.texture_extract = TextureMap(scale=4)
        self.k = ConvBNReLU(1, 1, 1, 1, 0, has_bn=False, has_relu=False)
        self.q = ConvBNReLU(1, 1, 1, 1, 0, has_bn=False, has_relu=False)
        self.v = ConvBNReLU(128, 32, 3, 1, 1, has_bn=False, has_relu=False)

        self.out = ConvBNReLU(64 + 32, 32, 1, 1, 0)

    def forward(self, x):
        resident = x
        txt, x = self.texture_extract(x)
        txt = txt.unsqueeze(dim=1)
        k = self.k(txt)
        q = self.q(txt)
        v = self.v(x)
        N, C, H, W = v.shape
        k = k.reshape(N, H * W, -1)
        q = q.reshape(N, -1, H * W)
        w = torch.bmm(k, q)
        w = F.softmax(w, dim=-1)

        v = v.reshape(N, C, H * W)
        f = torch.bmm(v, w)
        f = f.reshape(N, C, H, W)
        # f = F.interpolate(f, scale_factor=2)
        # out = resident + f
        out = self.out(torch.cat((f, resident), dim=1))

        return out


class ConvBNReLU(nn.Module):
    '''Module for the Conv-BN-ReLU tuple.'''

    def __init__(self, c_in, c_out, kernel_size=3, stride=1, padding=1, dilation=1, group=1,
                 has_bn=True, has_relu=True, mode='2d'):
        super(ConvBNReLU, self).__init__()
        self.has_bn = has_bn
        self.has_relu = has_relu
        if mode == '2d':
            self.conv = nn.Conv2d(
                c_in, c_out, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=False, groups=group)
            norm_layer = nn.BatchNorm2d
        elif mode == '1d':
            self.conv = nn.Conv1d(
                c_in, c_out, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=False, groups=group)
            norm_layer = nn.BatchNorm1d
        if self.has_bn:
            self.bn = norm_layer(c_out)
        if self.has_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        return x

class DetailBranch(nn.Module):

    def __init__(self):
        super(DetailBranch, self).__init__()

        # self.S0 = nn.Sequential(
        #     ConvBNReLU(3, 32, 3, stride=1),
        #     ConvBNReLU(32, 16, 3, stride=1),
        #     ConvBNReLU(16, 64, 3, stride=1),
        #     nn.PixelShuffle(2),
        #     ConvBNReLU(16, 32, 3, stride=2)
        # )

        self.S1 = nn.Sequential(
            ConvBNReLU(3, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
        )

        self.S2 = nn.Sequential(
            ConvBNReLU(64, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
            ConvBNReLU(64, 64, 3, stride=1),
        )

        self.S3 = nn.Sequential(
            ConvBNReLU(64, 128, 3, stride=2),
            ConvBNReLU(128, 128, 3, stride=1),
            ConvBNReLU(128, 128, 3, stride=1),
        )

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.extract_edge1 = nn.Sequential(
            ConvBNReLU(64, 32, 3, 1, 1, has_relu=False),
            nn.Conv2d(32, 1, (1, 1), padding=0))

        self.up2 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.extract_edge2 = nn.Sequential(
            ConvBNReLU(64, 32, 3, 1, 1, has_relu=False),
            nn.Conv2d(32, 1, (1, 1), padding=0))

        self.up3 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)
        self.extract_edge3 = nn.Sequential(
            ConvBNReLU(128, 32, 3, 1, 1, has_relu=False),
            nn.Conv2d(32, 1, (1, 1), padding=0))

    def forward(self, x):
        feat = x
        # feat = self.S0(feat)

        feat = self.S1(feat)
        edge1 = self.up1(feat)
        edge1 = self.extract_edge1(edge1)
        feat_supply = feat

        feat = self.S2(feat)
        edge2 = self.up2(feat)
        edge2 = self.extract_edge2(edge2)

        feat = self.S3(feat)
        edge3 = self.up3(feat)
        edge3 = self.extract_edge3(edge3)

        return feat, feat_supply, [edge1, edge2, edge3]


class StemBlock(nn.Module):

    def __init__(self):
        super(StemBlock, self).__init__()
        self.conv = ConvBNReLU(3, 16, 3, stride=2)
        self.left = nn.Sequential(
            ConvBNReLU(16, 8, 1, stride=1, padding=0),
            ConvBNReLU(8, 16, 3, stride=2),
        )
        self.right = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.fuse = ConvBNReLU(32, 16, 3, stride=1)

    def forward(self, x):
        feat = self.conv(x)
        feat_left = self.left(feat)
        feat_right = self.right(feat)
        feat = torch.cat([feat_left, feat_right], dim=1)
        feat = self.fuse(feat)
        return feat


class CEBlock(nn.Module):

    def __init__(self):
        super(CEBlock, self).__init__()
        self.bn = nn.BatchNorm2d(128)
        self.conv_gap = ConvBNReLU(128, 128, 1, stride=1, padding=0)
        #TODO: in paper here is naive conv2d, no bn-relu
        self.conv_last = ConvBNReLU(128, 128, 3, stride=1)

    def forward(self, x):
        feat = torch.mean(x, dim=(2, 3), keepdim=True)
        feat = self.bn(feat)
        feat = self.conv_gap(feat)
        feat = feat + x
        feat = self.conv_last(feat)
        return feat


class GELayerS1(nn.Module):

    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(GELayerS1, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(
                in_chan, mid_chan, kernel_size=3, stride=1,
                padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True), # not shown in paper
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.conv2[1].last_bn = True
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.conv2(feat)
        feat = feat + x
        feat = self.relu(feat)
        return feat


class GELayerS2(nn.Module):

    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(GELayerS2, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(
                in_chan, mid_chan, kernel_size=3, stride=2,
                padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
        )
        self.dwconv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, mid_chan, kernel_size=3, stride=1,
                padding=1, groups=mid_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True), # not shown in paper
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.conv2[1].last_bn = True
        self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_chan, in_chan, kernel_size=3, stride=2,
                    padding=1, groups=in_chan, bias=False),
                nn.BatchNorm2d(in_chan),
                nn.Conv2d(
                    in_chan, out_chan, kernel_size=1, stride=1,
                    padding=0, bias=False),
                nn.BatchNorm2d(out_chan),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv1(feat)
        feat = self.dwconv2(feat)
        feat = self.conv2(feat)
        shortcut = self.shortcut(x)
        feat = feat + shortcut
        feat = self.relu(feat)
        return feat


class SegmentBranch(nn.Module):

    def __init__(self):
        super(SegmentBranch, self).__init__()
        self.S1S2 = StemBlock()
        self.S3 = nn.Sequential(
            GELayerS2(16, 32),
            GELayerS1(32, 32),
        )

        # self.S3_enhance = nn.Sequential(ConvBNReLU(32, 64, 3, 1, 1),
        #                                 TEMDistinct())
        # self.S3_fuse = ConvBNReLU(64, 32, 3, stride=1)

        self.S4 = nn.Sequential(
            GELayerS2(32, 64),
            GELayerS1(64, 64),
        )
        self.S5_4 = nn.Sequential(
            GELayerS2(64, 128),
            GELayerS1(128, 128),
            GELayerS1(128, 128),
            GELayerS1(128, 128),
        )
        self.S5_5 = CEBlock()

    def forward(self, x):
        feat2 = self.S1S2(x)
        feat3 = self.S3(feat2)
        # enhance = self.S3_enhance(feat3)
        # feat4 = torch.cat((feat3, enhance), dim=1)
        # feat4 = self.S3_fuse(feat4)
        feat4 = self.S4(feat3)
        feat5_4 = self.S5_4(feat4)
        feat5_5 = self.S5_5(feat5_4)
        return feat2, feat3, feat4, feat5_4, feat5_5


class BGALayer(nn.Module):

    def __init__(self):
        super(BGALayer, self).__init__()
        self.left1 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                128, 128, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        self.left2 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=2,
                padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        )
        self.right1 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.right2 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                128, 128, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        ##TODO: does this really has no relu?
        self.conv = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), # not shown in paper
        )

    def forward(self, x_d, x_s):
        dsize = x_d.size()[2:]
        left1 = self.left1(x_d)
        left2 = self.left2(x_d)
        right1 = self.right1(x_s)
        right2 = self.right2(x_s)
        right1 = F.interpolate(
            right1, size=dsize, mode='bilinear', align_corners=True)
        left = left1 * torch.sigmoid(right1)
        right = left2 * torch.sigmoid(right2)
        right = F.interpolate(
            right, size=dsize, mode='bilinear', align_corners=True)
        out = self.conv(left + right)
        return out



class SegmentHead(nn.Module):

    def __init__(self, in_chan, mid_chan, n_classes):
        super(SegmentHead, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, 3, stride=1)
        self.drop = nn.Dropout(0.1)
        self.conv_out = nn.Conv2d(
                mid_chan, n_classes, kernel_size=1, stride=1,
                padding=0, bias=True)

    def forward(self, x, size=None):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        if not size is None:
            feat = F.interpolate(feat, size=size,
                mode='bilinear', align_corners=True)
        return feat


class WZCNet(nn.Module):

    def __init__(self, n_classes):
        super(WZCNet, self).__init__()
        self.detail = DetailBranch()
        self.segment = SegmentBranch()
        self.bga = BGALayer()

        ## TODO: what is the number of mid chan ?
        self.head = SegmentHead(128, 1024, 2)
        # self.head = SegmentHead(128, 1024, 64)
        # self.head2 = SegmentHead(64 + 64, 256, n_classes)
        self.aux2 = SegmentHead(16, 128, n_classes)
        self.aux3 = SegmentHead(32, 128, n_classes)
        self.aux4 = SegmentHead(64, 128, n_classes)
        self.aux5_4 = SegmentHead(128, 128, n_classes)

        self.init_weights()

    def forward(self, x):
        size = x.size()[2:]
        feat_d, feat_supply, edge = self.detail(x)
        size_supply = feat_supply.size()[2:]
        feat2, feat3, feat4, feat5_4, feat_s = self.segment(x)
        feat_head = self.bga(feat_d, feat_s)

        # logits = self.head(feat_head, size_supply)
        # logits = self.head2(torch.cat((logits, feat_supply), dim=1), size=size)

        logits = self.head(feat_head, size)
        logits_aux2 = self.aux2(feat2, size)
        logits_aux3 = self.aux3(feat3, size)
        logits_aux4 = self.aux4(feat4, size)
        logits_aux5_4 = self.aux5_4(feat5_4, size)

        return [logits, logits_aux2, logits_aux3, logits_aux4, logits_aux5_4, edge]

    def init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if not module.bias is None: nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                if hasattr(module, 'last_bn') and module.last_bn:
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)


if __name__ == '__main__':
    from thop import profile

    input = torch.rand((1, 3, 512, 512)).cuda()
    model = WZCNet(2).eval().cuda()
    flops, params = profile(model, inputs=(input,))
    print("Number of FLOPS: %.2fG" % (flops / 1e9))
    print("Number of parameter: %.2fM" % (params / 1e6))

    total = sum([param.nelement() for param in model.parameters()])

    print("Number of parameter: %.2fM" % (total / 1e6))