"""AlexNet
Reference:
    - [ ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
"""

# import pkgs
import torch
import torch.nn as nn

# AlexNet model
class AlexNet(nn.Module):
    def __init__(self,num_classes:int=1000,init_weights:bool=False):
        super(AlexNet,self).__init__()
        # conv layers
        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(3,64,kernel_size=11,stride=4,padding=2),
            nn.ReLU(inplace=True),  
            nn.MaxPool2d(kernel_size=3,stride=2),
            # conv2
            nn.Conv2d(64,192,kernel_size=5,stride=1,padding=2),
            nn.ReLU(inplace=True),  
            nn.MaxPool2d(kernel_size=3,stride=2),
            # conv3
            nn.Conv2d(192,384,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),  
            # conv4
            nn.Conv2d(384,256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True), 
            # conv5
            nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),  
            nn.MaxPool2d(kernel_size=3,stride=2), 
        )
        # avg pool layer
        self.avgpool = nn.AdaptiveAvgPool2d((6,6))  #将256个NxN的特征图变成256个6x6的特征图

        # fc layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256*6*6,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5).float(),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,num_classes)
        )

        if init_weights:
            self._initialize_weights()

    def forward(self,x):
        x = self.features(x)
        x = torch.flatten(self.avgpool(x),1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    
    from torchsummary import summary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    # from torchvision.models import alexnet
    # model = alexnet().to(device)
    # summary(model,(3,227,227))

    """
    Total params: 61,100,840
    Trainable params: 61,100,840
    Non-trainable params: 0
    """

    model = AlexNet(num_classes=1000).to(device)
    summary(model,(3,227,227))

    """
    Total params: 61,100,840
    Trainable params: 61,100,840
    Non-trainable params: 0
    """

    x = torch.ones([1,3,224,224])
    res = model(x.to(device))
    print(res.shape) # torch.Size([1, 1000])