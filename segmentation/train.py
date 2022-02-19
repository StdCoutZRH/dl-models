import torch,random
import numpy as np
import torch.nn as nn
from torch import optim
import torch.backends.cudnn  as cudnn
from tqdm import tqdm

from datasets.whu import WHU
from models.my_model import MobileSegViT
from utils.distributed_utils import *
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

def set_seed(seed=233, rank=-1, cuda_deterministic=True):
    # 必须禁用模型初始化中的任何随机性。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if rank != -1:
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True
        cudnn.deterministic = False
    cudnn.enabled = True

checkpoints = 'checkpoints/'

def create_dataloader(data_root, batch_size=16, train=True):
    mode = "train" if train else "val"
    dataset = WHU(data_root=data_root, mode=mode)
    dataloader = DataLoader(dataset, batch_size=batch_size,shuffle=train,  pin_memory=True)
    print(len(dataset), len(dataloader))
    return dataloader


def create_lr_scheduler(optimizer, num_step: int, epochs: int, warmup=True, warmup_epochs=1, warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    def f(x):
        """
        根据step数返回一个学习率倍率因子，注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def train(model, device):

    # set model to device
    device = device
    print(f'Using device {device}')
    model.to(device=device)

    # get dataloader
    train_root = "../../data/WHU/train/"
    val_root = "../../data/WHU/test/"
    train_loader = create_dataloader(data_root=train_root, train=True, batch_size=16)
    valid_loader = create_dataloader(data_root=val_root, train=False, batch_size=16)


    #rank = -1
    # DDP init
    #if rank != -1:
    #    torch.cuda.set_device(rank)
    #    dist.init_process_group(
     #       backend='nccl',
            # init_method='tcp://127.0.0.1:43260',
            # world_size=torch.cuda.device_count(),
     #       rank=rank)
    set_seed(223, -1)

    #if rank != -1:
    #    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
    #    model = DDP(model,
    #                device_ids=[-1],
    #                output_device=-1,
    #                find_unused_parameters=True
    #                )
    #else:
    #    model = torch.nn.DataParallel(model, device_ids=['0','1']).cuda()
    
    # tensorboard
    writer = SummaryWriter(log_dir='tensorboard/')
    global_step = 0

    # 定义优化器和loss
    optimizer = optim.RMSprop(
        model.parameters(), lr=0.001, weight_decay=1e-4, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), 200, warmup=True)
    best_miou = 0
    print("Start training.")

    for epoch in range(300):
        model.train()
        epoch_loss = 0

        # 一个batch一个batch地训练
        for images, masks in tqdm(train_loader):  # 获取图片和masks
            images = images.to(device=device, dtype=torch.float32)
            true_masks = masks.to(device=device, dtype=torch.float32)

            pred_masks = model(images)  # 正向传递获得预测mask
            loss = criterion(pred_masks, true_masks)  # 计算loss
            epoch_loss += loss.item()  # 累加epoch loss

            # loss是一个batch的loss，epoch_loss才是一个epoch的loss
            # writer.add_scalar('Loss/train', loss.item(), global_step)
            # pbar.set_postfix(**{'loss (batch)': loss.item()})

            optimizer.zero_grad()
            loss.backward()  # 反向传播
            nn.utils.clip_grad_value_(model.parameters(), 0.1)
            optimizer.step()  # 优化
            lr_scheduler.step()

        # 一个epoch评估一次
        model.eval()
        train_confmat = ConfusionMatrix(2)
        with torch.no_grad():
            for image, mask in tqdm(train_loader):
                image, mask = image.to(device=device, dtype=torch.float32), mask.to(
                    device=device, dtype=torch.float32)
                pred_mask = model(image)  # 正向传递获得预测mask
                train_confmat.update(mask.argmax(1).flatten(),
                               pred_mask.argmax(1).flatten())

            train_confmat.reduce_from_all_processes()
        train_iou,train_miou = train_confmat.get_metric()
        print(f"train:---iou:{train_iou}---miou:{train_miou}")
 
        test_confmat = ConfusionMatrix(2)
        with torch.no_grad():
            for image, mask in tqdm(valid_loader):
                image, mask = image.to(device=device, dtype=torch.float32), mask.to(
                    device=device, dtype=torch.float32)
                pred_mask = model(image)  # 正向传递获得预测mask
                test_confmat.update(mask.argmax(1).flatten(),
                               pred_mask.argmax(1).flatten())

            test_confmat.reduce_from_all_processes()

        test_iou,test_miou = test_confmat.get_metric()
        print(f"train:---iou:{test_iou}---miou:{test_miou}")

        
        lr = optimizer.param_groups[0]["lr"]
        print("ccurrent lr:",lr)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

        # 输出dice off 和loss
        print(f'After Epoch {epoch} , Loss:{epoch_loss}')
        writer.add_scalar('Dice/test', val_score, global_step)

        writer.add_images('images', images, global_step)
        # if model.n_classes == 1:
        #     writer.add_images('masks/true', true_masks, global_step)
        #     writer.add_images('masks/pred', torch.sigmoid(pred_masks) > 0.5, global_step)

        if test_miou > best_miou:   #只保存最好的模型
            best_miou = test_miou
            save_path = f"checkpoints/epoch_{epoch}_miou_{best_miou}.pth"  
            torch.save(model.state_dict(), save_path)
            print(f'Checkpoint {epoch} saved !')

if __name__ == '__main__':

    device = torch.device('cuda:0')
    model = MobileSegViT(2)
    train(model=model, device=device)