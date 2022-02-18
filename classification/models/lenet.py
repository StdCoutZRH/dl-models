"""LeNet
Reference:
    - [Gradient-Based Learning Applied to Document Recognition](https://ieeexplore.ieee.org/document/726791)
"""

# import pkgs
import torch
import torch.nn as nn

# LeNet model
class LeNet(nn.Module):
    def __init__(self,num_classes:int=10,init_weights:bool=False):
        super(LeNet,self).__init__()
        # conv layers
        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(1,6,3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            # conv2
            nn.Conv2d(6,16,3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)
        )
        # fc layers
        self.classifier = nn.Sequential(
            nn.Linear(16*6*6,120),
            nn.ReLU(inplace=True),
            nn.Linear(120,84),
            nn.ReLU(inplace=True),
            nn.Linear(84,10),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self,x):
        B,C,H,W = x.shape
        assert H == 32 and W == 32,f"Input image size should be 32x32."
        x = self.features(x)
        x = torch.flatten(x,1)
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

    # # test assert input size
    # model = LeNet()
    # ret = model(torch.randn(1,1,32,32))   # B,C,H,W
    # print(ret)
    
    from torchsummary import summary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    model = LeNet(num_classes=10).to(device)
    summary(model,(1,32,32))

    """
    Total params: 81,194
    Trainable params: 81,194
    Non-trainable params: 0
    """

    x = torch.ones([1,1,32,32])
    res = model(x.to(device))
    print(res.shape)    # torch.Size([1, 10])