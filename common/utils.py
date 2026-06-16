r""" Helper functions """
import random
import torch.nn.functional as F
import torch
import numpy as np


def fix_randseed(seed):
    r""" Set random seeds for reproducibility """
    if seed is None:
        seed = int(random.random() * 1e5)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mean(x):
    return sum(x) / len(x) if len(x) > 0 else 0.0


def to_cuda(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.cuda()
    return batch


def to_cpu(tensor):
    return tensor.detach().clone().cpu()
def resize_batch(batch, size):
    """
    调整批次中所有图像和掩码到指定尺寸
    """
    resized_batch = {}
    for key, value in batch.items():
        if key in ['query_img', 'support_imgs', 'support_masks', 'history_mask']:
            if isinstance(value, torch.Tensor):
                # 调整图像和掩码
                resized = F.interpolate(value, size=(size, size), mode='bilinear', align_corners=True)
                resized_batch[key] = resized
            else:
                # 对于非张量数据，保持原样
                resized_batch[key] = value
        elif key == 'org_query_imsize':
            # 记录原始尺寸
            resized_batch[key] = value
        elif key == 'org_query_img':
            # 保存原始图像用于CRF
            resized_batch[key] = value
        else:
            # 其他数据保持不变
            resized_batch[key] = value
    return resized_batch
