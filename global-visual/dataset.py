import os
import random

import h5py
import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class Feature_Dataset(Dataset):
    def __init__(self, h5_path, dataset="Shanghai", test=False):
        super(Feature_Dataset, self).__init__()
        self.h5_path = h5py.File(h5_path, "r")
        self.mode = "test" if test else "train"
        self.dataset = dataset
        self.videos = sorted(self.h5_path.keys())
        self.psts = []
        self.prsts = []
        self.ftrs = []
        self.all_seqs = []
        self.labels = []
        self.idxes = []
        self.load_feat()

    def load_feat(self):
        if self.mode == "train":
            for video in self.videos:
                pst, prst, ftr = np.array(self.h5_path[f"{video}"])

                self.psts.append(pst)
                self.prsts.append(prst)
                self.ftrs.append(ftr)
                random_seq = list(range(len(prst)))
                random.shuffle(random_seq)
                self.all_seqs.append(random_seq)

        else:
            for video in self.videos:
                pst, prst, ftr = np.array(self.h5_path[f"{video}"])
                self.psts.append(pst)
                self.prsts.append(prst)
                self.ftrs.append(ftr)

                label = np.load(
                    f"../labels/{self.dataset}/"
                    + f"{video.split('.')[0]}.npy"
                )

                self.labels.append(label)

    def __getitem__(self, idx):
        pst = torch.from_numpy(self.psts[idx]).float()
        prst = torch.from_numpy(self.prsts[idx]).float()
        ftr = torch.from_numpy(self.ftrs[idx]).float()
        if self.mode == "test":
            label = self.labels[idx]
            return pst, prst, ftr, label
        
        start = self.all_seqs[idx].pop()
        if len(self.all_seqs[idx]) == 0:
            self.all_seqs[idx] = list(range(len(prst)))
            random.shuffle(self.all_seqs[idx])

        return pst[start], prst[start], ftr[start]

    def __len__(self):
        return len(self.psts)
