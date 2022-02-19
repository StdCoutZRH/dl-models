from models.wzcnet import WZCNet
from utils.show_model_in_tensorboard import show_model_in_tensorboard

import numpy as np

if __name__ == '__main__':
    
    mask =np.array([0,1,0,0,1])
    mask = np.eye(2)[mask.reshape([-1])]
    mask = mask.reshape(1, 5, 2)  # （512,512,2）
    print(mask.shape)