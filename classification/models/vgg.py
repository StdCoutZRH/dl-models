"""VGG
References:
    - [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
"""

# import pkgs
import torch
import torch.nn as nn

# VGG
class VGG(nn.Module):
    def __init__(self,model_name:str='vgg16',num_classes:int=1000,init_weights:bool=False):
        super(VGG,self).__init__()
        
        self.cfgs = {
            # 数字代表卷积核个数，M代表Maxpooling
            'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }
        assert model_name in self.cfgs,"Error:model_name should be 'vgg16' or 'vgg19'!"
        
        # conv layers
        self.features = self.build_featurs(model_name) # featurs是构建卷积层的方法函数
        # fc layers
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096,4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096,num_classes)
        )

    def forward(self,x):
        x = torch.flatten(self.features(x),start_dim=1) # Nx3x224x224 -> Nx512x7x7 -> Nx512*7*7
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias,0)
    
    def build_featurs(self,model_name):
        cfg = self.cfgs[model_name]
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M': # 添加最大池化层
                layers+=[nn.MaxPool2d(kernel_size=2,stride=2)]
            else: # 添加卷积层
                layers+=[nn.Conv2d(in_channels,v,kernel_size=3,padding=1),nn.ReLU(True)]
                in_channels = v
        
        #最后加上自适应平均池化，这样模型可以适应任意input_size
        layers += [nn.AdaptiveAvgPool2d((7,7))]   
        return nn.Sequential(*layers)

if __name__ == '__main__':

    # 2种方法查看参数量

    # 1.直接统计然后打印
    #model = VGG(model_name='vgg16')
    #total_params = sum(p.numel() for p in model.parameters())
    #print(f'{total_params:,} total parameters.')
    #total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print(f'{total_trainable_params:,} training parameters.')   

    '''
    vgg16:
    -138,357,544 total parameters.
    -138,357,544 training parameters.

    vgg19:
    -143,667,240 total parameters.
    -143,667,240 training parameters.
    '''

    # 2.使用torchsummery这个库
    from torchsummary import summary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = VGG().to(device)
    summary(model,(3,224,224))

    x = torch.ones([1,3,224,224])
    res = model(x.to(device))
    print(res.shape) # torch.Size([1, 1000])