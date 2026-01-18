import numpy as np
import torch
import random
import errno
import os
import sys
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset,Dataset

from utils.cutmix import  CutMix
from utils.random_crop import RandomCrop
from utils.random_erasing import RandomErasing


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def set_save_path(father_path, args):
    father_path = os.path.join(father_path, '{}'.format(time.strftime("%m_%d_%H_%M")))
    mkdir(father_path)
    args.log_path = father_path
    args.model_path = father_path
    args.result_path = father_path
    args.spatial_adj_path = father_path
    args.time_adj_path = father_path
    args.tensorboard_path = father_path
    return args


def sliding_window_eeg(data,label,window_size,stride):
    # print(data.shape)
    trails = data.shape[0]
    num_channels = data.shape[1]
    num_samples = data.shape[2]
    num_segments = (num_samples - window_size)//stride + 1
    segments = np.zeros((num_segments,trails,num_channels,window_size),dtype=np.float64)

    for i in range(trails):
        for j in range(num_segments):
            start = j * stride
            end = start + window_size
            segments[j][i] = data[i,:,start:end]
    
    return segments,label



EOS = 1e-10
def normalize(adj):
    adj = F.relu(adj)
    inv_sqrt_degree = 1. / (torch.sqrt(torch.sum(adj,dim=-1,keepdim=False)) + EOS)
    return inv_sqrt_degree[:,None]*adj*inv_sqrt_degree[None,:]




def save(checkpoints, save_path):
    torch.save(checkpoints, save_path)

def accuracy(output, target, topk=(1,)):
    shape = None
    if 2 == len(target.size()):
        shape = target.size()
        target = target.view(target.size(0))
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    ret = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
        ret.append(correct_k.mul_(1. / batch_size))
    if shape:
        target = target.view(shape)
    return ret


class Logger(object):
    def __init__(self, fpath):
        self.console = sys.stdout
        self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, *args):
        for t in self.transforms:
            x = t(x, *args)
        return x

def build_tranforms():
    return Compose([
        RandomCrop(1125),
        CutMix(),  
        # RandomErasing(),
    ])

class EEGDataSet(Dataset):
    def __init__(self,data,label):
        self.label = label
        self.data = data
    
    def __len__(self):
        return self.data.shape[1]
    
    def __getitem__(self, index):
        data = self.data[:,index]
        label = self.label[index]
        return data,label

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


