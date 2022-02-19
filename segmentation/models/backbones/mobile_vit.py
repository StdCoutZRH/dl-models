import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from vit import TransformerEncoder
from mobilenetv2 import *

class MobileVitBlock(nn.Module):
    def __init__(self,
            in_channels=3,dim=512,kernel_size=3,    
            path_size=7,heads=8,depth=3,mlp_dim=1024,blocks=2,patch_h=8,patch_w=8
            ):
        super(MobileVitBlock, self).__init__()
        self.blocks = blocks
        self.in_channels = in_channels
        self.dim = dim
        self.heads = heads
        self.kernel_size=kernel_size
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h

        # local representations
        self.local_rep = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,kernel_size=kernel_size,padding=kernel_size//2),
            nn.Conv2d(in_channels,dim,kernel_size=1)
        )
        
        # transformers
        self.transformers = TransformerEncoder(self.dim,self.heads)

    def unfold(self,x): # x:[B, C, H, W]
        pw,ph = self.patch_h,self.patch_w
        patch_area = self.patch_area
        batch_size,in_channels,origin_h,origin_w = x.shape

        new_h = int(math.ceil(origin_h / ph) * ph)
        new_w = int(math.ceil(origin_w / pw) * pw)

        interpolate = False
        if new_w != origin_w or new_h != origin_h:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
            interpolate = True

        # w和h方向上的patches数以及总的patches数
        num_patch_w = new_w // pw # n_w
        num_patch_h = new_h // ph # n_h
        num_patches = num_patch_h * num_patch_w # N

        # [B, C, H, W] --> [B * C * n_h, p_h, n_w, p_w]
        reshaped_x = x.reshape(batch_size * in_channels * num_patch_h, ph, num_patch_w, pw)
        # [B * C * n_h, p_h, n_w, p_w] --> [B * C * n_h, n_w, p_h, p_w]
        transposed_x = reshaped_x.transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] --> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        reshaped_x = transposed_x.reshape(batch_size, in_channels, num_patches, patch_area)
        # [B, C, N, P] --> [B, P, N, C]
        transposed_x = reshaped_x.transpose(1, 3)
        # [B, P, N, C] --> [BP, N, C]
        patches = transposed_x.reshape(batch_size * patch_area, num_patches, -1)
        info_dict = {
            "orig_size": (origin_h, origin_w),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h
        }
        return patches, info_dict


    def fold(self, patches, info_dict):
        n_dim = patches.dim()
        assert n_dim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(patches.shape)
        # [BP, N, C] --> [B, P, N, C]
        patches = patches.contiguous().view(info_dict["batch_size"], self.patch_area, info_dict["total_patches"], -1)

        batch_size, pixels, num_patches, channels = patches.size()
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]

        # [B, P, N, C] --> [B, C, N, P]
        patches = patches.transpose(1, 3)

        # [B, C, N, P] --> [B*C*n_h, n_w, p_h, p_w]
        feature_map = patches.reshape(batch_size * channels * num_patch_h, num_patch_w, self.patch_h, self.patch_w)
        # [B*C*n_h, n_w, p_h, p_w] --> [B*C*n_h, p_h, n_w, p_w]
        feature_map = feature_map.transpose(1, 2)
        # [B*C*n_h, p_h, n_w, p_w] --> [B, C, H, W]
        feature_map = feature_map.reshape(batch_size, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w)
        if info_dict["interpolate"]:
            feature_map = F.interpolate(feature_map, size=info_dict["orig_size"], mode="bilinear", align_corners=False)
        return feature_map

    def funsion(self,x,y):
        x = nn.Conv2d(self.dim,self.in_channels,kernel_size=1)(x)   # H W C
        cat = torch.cat((x, y), dim=1)
        out = nn.Conv2d(2*self.in_channels,self.in_channels,kernel_size=self.kernel_size,padding=self.kernel_size//2)(cat)
        return out

    
    def forward(self,x):
        res = x

        # local rep
        x = self.local_rep(x)   # 1 512 32 32

        # convert feature map to patches
        patches, info_dict = self.unfold(x) # 64 16 512

        # learn global representations
        for _ in range(self.blocks):
            patches = self.transformers(patches)   # 64 16 512

        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        fm = self.fold(patches=patches, info_dict=info_dict)

        return self.funsion(fm,res)

class MobileViT(nn.Module):
    def __init__(self,
            image_size=224,in_channels=3,
            dims=[64,80,96],blocks=[2,4,3],channels=[16, 16, 24, 24, 48, 64, 80, 320],
            num_classes=1000,
            ):
        super(MobileViT, self).__init__()
        # image
        self.image_size = image_size
        self.in_channels = in_channels
        self.channels = channels
        self.classes = num_classes
        # transformers
        self.dims = dims
        self.blocks = blocks

        # conv layers
        self.conv_layers =[
            # conv+MV2*4
            nn.Sequential(
            ConvBNReLU(in_channels=self.in_channels,out_channels=self.channels[0],stride=2), 
            InvertedResidual(in_channels=self.channels[0],out_channels=self.channels[1],stride=1),
            InvertedResidual(in_channels=self.channels[1],out_channels=self.channels[2],stride=2),
            InvertedResidual(in_channels=self.channels[2],out_channels=self.channels[3],stride=1),
            InvertedResidual(in_channels=self.channels[3],out_channels=self.channels[4],stride=2)),
            
            # MV2*1
            InvertedResidual(in_channels=self.channels[4],out_channels=self.channels[5],stride=2),
            InvertedResidual(in_channels=self.channels[5],out_channels=self.channels[6],stride=2)
        ] 

        # mobile vit blocks
        self.mvt_blocks=[
            MobileVitBlock(in_channels=self.channels[4],dim=self.dims[0],blocks=self.blocks[0]),
            MobileVitBlock(in_channels=self.channels[5],dim=self.dims[1],blocks=self.blocks[1]),
            MobileVitBlock(in_channels=self.channels[6],dim=self.dims[2],blocks=self.blocks[2]),
        ]

    def forward(self,x):
        for i in range(3):
            print(i)
            x = self.conv_layers[i](x)
            print("conv:",x.shape)
            x = self.mvt_blocks[i](x)
            print("msa:",x.shape)
        x = ConvBNReLU(in_channels = self.channels[-2],out_channels=self.channels[-1],kernel_size=1 )(x)
        x = nn.AvgPool2d(self.image_size//32,1)(x)
        x = x.view(x.shape[0],-1)
        x = nn.Linear(in_features=self.channels[-1],out_features=self.classes,bias=True)(x)
        return x

def mobilevit_xxs(image_size):
    dims=[64,80,96]
    channels= [16, 16, 24, 24, 48, 64, 80, 320]
    return MobileViT(image_size=image_size,dims=dims,channels=channels,num_classes=1000)

def mobilevit_xs(image_size):
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 80, 96, 384]
    return MobileViT(image_size=image_size,dims=dims,channels=channels,num_classes=1000)

def mobilevit_s(image_size):
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 128, 160, 640]
    return MobileViT(image_size=image_size,dims=dims,channels=channels,num_classes=1000)


if __name__ == '__main__':

    x = torch.ones([1,3,512,512])
    model = mobilevit_xxs(512)
    res = model(x)

    print(res.shape)