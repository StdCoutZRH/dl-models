from .backbones.vit_swin import *
import math
from einops import rearrange


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True))
    def forward(self,x):
        return self.layer(x)

class LocalBranch(nn.Module):
    def __init__(self) :
        super(LocalBranch,self).__init__()

        self.step1 = nn.Sequential(ConvBNReLU(3,64,stride=2),ConvBNReLU(64,64,stride=1))
        self.step2 = nn.Sequential(ConvBNReLU(64,128,stride=2),ConvBNReLU(128,128,stride=1),ConvBNReLU(128,128,stride=1))
        self.step3 = nn.Sequential(ConvBNReLU(128,256,stride=2),ConvBNReLU(256,256,stride=1),ConvBNReLU(256,256,stride=1))

        self.init_weights()

    def forward(self,x):
        # [1, 3, 512, 512]
        x_256 = self.step1(x)   # [1, 64, 256, 256]
        x_128 = self.step2(x_256)   # [1, 128, 128, 128]
        x_64 = self.step3(x_128)   # [1, 256, 64, 64]
        return x_256,x_128,x_64

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

class GlobalBranch(nn.Module):
    def __init__(self, patch_size=4, in_chans=3,
                 embed_dim=96, depths=(2, 2, 2), num_heads=(6, 12, 24),
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super(GlobalBranch,self).__init__()
        self.num_layers = len(depths)   #几个stage
        self.embed_dim = embed_dim  # 每个token的dim
        self.patch_norm = patch_norm
        # stage4输出特征矩阵的channels
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1)) #8
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # 注意这里构建的stage和论文图中有些差异
            # 这里的stage不包含该stage的patch_merging层，包含的是下个stage的
            layers = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer < self.num_layers) else None,
                                use_checkpoint=use_checkpoint)
            self.layers.append(layers)

        norms = [192,384,768]
        self.norm_layers = nn.ModuleList()
        for num in norms:
            self.norm_layers.append(norm_layer(num))
        self.apply(self._init_weights)


    def patches_to_fm(self, x): # patches to feature map     
        # patches [1,256,768]
        B,N,C = x.shape
    
        # [B, N, C] --> [B, H,W,C]
        W = int(math.sqrt(N))
        H = W
        x = x.reshape(B,H,W,C)    #[1,16,16,768]

        # [1,16,16,768]-> [1,64,64,48]
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=4, p2=4, c=C//16)
    
        #  [B,H,W,C]--> [B,C,H,W] [1,48,64,64]
        feature_map = x.permute(0,3,1,2)
        return feature_map

    def forward(self, x):
        # x: [B, L, C] 512
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)    # [1,16384,96] 1/4 128*128

        feats = []
        index = 0
        for layer in self.layers:
            x, H, W = layer(x, H, W)
            x_norm = self.norm_layers[index](x)
            fm = self.patches_to_fm(x_norm)
            index+=1
            feats.append(fm)
            # [1,4096,192]  1/8 64*64
            # [1,1024,384]  1/16 32*32
            # [1,256,768]   1/32 16*16


        #x = self.norm(x)  # [B, L, C]   [1,256,768]
        #x = self.avgpool(x.transpose(1, 2))  # [B, C, 1]
        #x = torch.flatten(x, 1)
        #x = self.head(x)
        return feats
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

# upsample
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        # conv-bn-relu-conv-bn-relu
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)

class UpBilinear(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UpSample2x(nn.Module):
    """
    HW翻倍，C不变
    """
    def __init__(self, in_channels, factor=2):
        super(UpSample2x, self).__init__()
        out_channels = in_channels * factor * factor
        self.proj = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.up = nn.PixelShuffle(factor)
        self.init_weight()

    def forward(self, x):
        feat = self.proj(x)
        feat = self.up(feat)
        return feat

    def init_weight(self):
        nn.init.xavier_normal_(self.proj.weight, gain=1.)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MobileSegViT(nn.Module):
    def __init__(self,num_classes) -> None:
        super().__init__()

        self.global_branch = GlobalBranch()
        self.local_branch = LocalBranch()

        self.button = SELayer(48+256)


        self.up1 =  UpSample2x(304)
        self.up2 =  UpSample2x(456)
        self.up3 =  UpSample2x(532)
        self.out_conv = nn.Conv2d(532, num_classes, kernel_size=1)

        #self.conv1 = DoubleConv(304, 256, 304 // 2)
        #self.up2 =  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #self.conv2 = DoubleConv(384, 128, 256 // 2)
        #self.up3 =  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #self.conv3 = DoubleConv(192, 64, 128 // 2) 

        
 
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


    

    def forward(self,x):
        g_256,g_128,g_64 = self.global_branch(x)   #12 24 48
        l_256,l_128,l_64 = self.local_branch(x)  #64 128 256

        f_256 = torch.cat([g_256,l_256],dim=1)
        f_128 = torch.cat([g_128,l_128],dim=1)
        f_64 = torch.cat([g_64,l_64],dim=1)

        f_64=self.button(f_64)  #[1, 304, 64, 64]
        
        up1 = self.up1(f_64)
        up1 = torch.cat([up1,f_128],dim=1)  #[1, 456, 128, 128]

        up2 = self.up2(up1)
        up2 = torch.cat([up2,f_256],dim=1)  #[1, 532, 256, 256]

        up3 = self.up3(up2)
        out = self.out_conv(up3)

        return out

def show_params(model):
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))
   
if __name__ == "__main__":

    #global_branch = GlobalBranch()
    #local_branch = LocalBranch()

    #show_params(global_branch) 
    #show_params(local_branch)

    #up = UpSample2x(256,2)

    se = SELayer(48+256)
    x = torch.ones([1,3,512,512])
    x_1 = torch.ones([1, 304, 64, 64])

    model = MobileSegViT(2)
    show_params(model)
    #show_params(model)
    res = model(x)

    print(res.shape)
