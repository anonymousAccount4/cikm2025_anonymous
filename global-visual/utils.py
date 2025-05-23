import random
from functools import partialmethod

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def init_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    cudnn.deterministic = False
    random.seed(seed)


def smooth_scores(scores_arr, sigma=7):
    scores_arr_ = scores_arr.copy()
    for s in range(len(scores_arr_)):
        for sig in range(1, sigma):
            scores_arr_[s] = gaussian_filter1d(scores_arr_[s], sigma=sig)
    return scores_arr_

def MinMaxNorm(x):
    x_ = x.copy()
    np_x = np.concatenate(x_)
    min = np_x.min()
    max = np_x.max()
    for i in range(len(x)):
        x_[i] = (x_[i] - min) / (max - min)
    return x_
