import cv2
import os
from torch.utils.data import Dataset
from tqdm import tqdm

from .my_transforms import *

class WHU(Dataset):
    def __init__(self, data_root: str, 
                        mode="train", 
                        num_classes=2,
                        mean=(0.336925, 0.362922, 0.339419), 
                        std=(0.144224, 0.138708, 0.140357)):
        """
        data_path:path of dataset
        """
        super(WHU, self).__init__()

        assert os.path.exists(data_root), f"path '{data_root}' does not exist."
        self.imgs_dir = os.path.join(data_root, "image/")  # image path
        self.masks_dir = os.path.join(data_root+"mask/")  # label path
        # save all image names
        self.image_names = [file for file in os.listdir(self.imgs_dir)]
        print(f'{mode}:Creating dataset with {len(self.image_names)} examples.')

        self.label_map = {0: 0, 255: 1}
        self.num_classes = num_classes
        self.mean = mean
        self.std = std
        self.mode = mode

    def __len__(self):
        return len(self.image_names)

    # 原先mask图片里是0和255的像素值，将其转化为0和1的类别值
    def convert_mask(self, mask, reverse=False):
        temp = mask.copy()
        if reverse:
            for v, k in self.label_map.items():
                mask[temp == k] = v
        else:
            for k, v in self.label_map.items():
                mask[temp == k] = v
        return mask

    def preprocess(self, image, mask):
        # 将mask从0和255转换为0和1
        mask = self.convert_mask(mask, False)   #(512*512,1)
        
        # 将每个位置像素的类别值进行one-hot编码，也就是 mask：(W*H,1)->(W*H,num_classes)
        mask = np.eye(self.num_classes)[mask.reshape([-1])]
        mask = mask.reshape(512, 512, 2)  #(512,512,2）

        if self.mode == "train":
            tfs = [
                RandomHorizontalFlip(0.5), RandomVerticalFlip(0.5),
                ToTensor(), Normalize(self.mean, self.std)
            ]
        else:
            tfs = [ToTensor(), Normalize(self.mean, self.std)]
        image, mask = Compose(tfs)(image, mask)
        return image, mask

    def __getitem__(self, index):
        # 获取image和mask的路径
        image_name = self.image_names[index]
        image_path = os.path.join(self.imgs_dir, image_name)
        mask_path = os.path.join(self.masks_dir, image_name)
        assert os.path.exists(image_path), f"file '{image_path}' does not exist."
        assert os.path.exists(mask_path), f"file '{mask_path}' does not exist."

        # 读取image和mask
        image = cv2.imread(image_path, 1)   # BGR [512,512,3]
        mask = cv2.imread(mask_path, 0)  # GRAY [512,512]
        image, mask = self.preprocess(image, mask)
        # After process
        #   image:[-1, 3, 512, 512]
        #   mask:[-1, 2, 512, 512]
        return image, mask


def get_mean_std(train_data_path: str, norm: bool = True):
    '''
    Compute mean and variance for training data
    return: (mean, std)
    '''

    files = os.listdir(train_data_path)
    print(f'Compute mean and std for {len(files)} images in training data.')

    mean = [0, 0, 0]
    std = [0, 0, 0]

    for file in tqdm(files):
        image = cv2.imread(train_data_path + file, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # 是否需要归一化选择除不除255
        rgb = np.asarray(rgb) / 255.0 if norm else np.asarray(rgb)

        for i in range(3):
            mean[i] += rgb[:, :, i].mean()
            std[i] += rgb[:, :, i].std()
    return np.asarray(mean) / len(files), np.asarray(std) / len(files)


# main
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    dataset = WHU('../../data/WHU/train/', mode="train")
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=True,  pin_memory=True)

    print(len(dataset), len(dataloader))
    for (image, label) in dataloader:
        # torch.Size([1,3,512,512]) torch.Size([1,2,512,512])
        print(image.size(), label.size())
        break