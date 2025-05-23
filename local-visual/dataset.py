import torch.nn as nn
import torch
import numpy as np
import os
from torch.utils.data.dataset import Dataset
import h5py
import pickle, random

def test(a, b):
    if len(a) > len(b):
        return 1
    elif len(a) == len(b):
        return 0
    else:
        return -1


class Feature_Track_Dataset(Dataset):
    def __init__(self, h5_path, dataset='shanghai', test=False):
        super().__init__()
        self.h5_path = h5_path
        self.h5py = h5py.File(self.h5_path, 'r')
        self.test = test
        self.dataset = dataset
        self.label_path = f'../labels/{self.dataset}/'
        self.videos = sorted(self.h5py.keys())
        self.clip_length = 4
        self.feats_pst = []
        self.feats_prst = []
        self.feats_ftr = []
        self.all_seqs = []
        self.labels = {}
        self.chunks = {}
        self.tracks_info = pickle.load(open(f'./features/{self.dataset}/track_info.pickle', 'rb'))
        self.infos = {}
        self.load_feat()

    def load_feat(self):
        if self.test == False:
            for video in self.videos:
                tracks = self.h5py[video].keys()
                psts, prsts, ftrs = [], [], []
                for track in tracks:
                    pst, prst, ftr = np.array(self.h5py[f"{video}/{track}"])
                    psts.append(pst)
                    prsts.append(prst)
                    ftrs.append(ftr)
                psts = np.concatenate(psts, axis=0)
                prsts = np.concatenate(prsts, axis=0)
                ftrs = np.concatenate(ftrs, axis=0)
                if psts.shape[0] < self.clip_length:
                    psts = np.pad(psts, ((0, self.clip_length - psts.shape[0]), (0, 0)), 'edge')
                    prsts = np.pad(prsts, ((0, self.clip_length - prsts.shape[0]), (0, 0)), 'edge')
                    ftrs = np.pad(ftrs, ((0, self.clip_length - ftrs.shape[0]), (0, 0)), 'edge')
                random_seq = list(range(len(psts) - self.clip_length + 1))
                random.shuffle(random_seq)
                self.all_seqs.append(random_seq)
                self.feats_pst.append(psts)
                self.feats_prst.append(prsts)
                self.feats_ftr.append(ftrs)


        else:
            sum_frame = 0
            for video in self.videos:
                tracks = self.h5py[video].keys()
                psts, prsts, ftrs = [], [], []
                for track in tracks:
                    pst, prst, ftr = np.array(self.h5py[f"{video}/{track}"])
                    psts.append(pst)
                    prsts.append(prst)
                    ftrs.append(ftr)
                
                    start, end = self.tracks_info[f'{video}/{track[:-4]}'].values()
                    if f'{video}' not in self.chunks:
                        self.chunks[f'{video}'] = [len(prst)]
                        self.infos[f'{video}'] = [(start, end)]
                    else:
                        self.chunks[f'{video}'].append(len(prst))
                        self.infos[f'{video}'].append((start, end))
                self.feats_pst.append(np.concatenate(psts, axis=0))   
                self.feats_prst.append(np.concatenate(prsts, axis=0))   
                self.feats_ftr.append(np.concatenate(ftrs, axis=0))   
            
            for video in sorted(os.listdir(self.label_path)):
                label = np.load(os.path.join(self.label_path, f"{video}"))
                self.labels[video.split('.')[0]] = label

    def __getitem__(self, idx):       
        pst = torch.from_numpy(self.feats_pst[idx]).float()
        prst = torch.from_numpy(self.feats_prst[idx]).float()
        ftr = torch.from_numpy(self.feats_ftr[idx]).float()

        if self.test:
            video = self.videos[idx]
            chunk, info = self.chunks[video], self.infos[video]
            return pst, prst, ftr, video, chunk, info
        
        start = self.all_seqs[idx].pop()
        if len(self.all_seqs[idx]) == 0:
            self.all_seqs[idx] = list(range(len(pst) - self.clip_length + 1))
            random.shuffle(self.all_seqs[idx])

        return pst[start:start+self.clip_length], prst[start:start+self.clip_length], ftr[start:start+self.clip_length]

    def __len__(self):
        return len(self.feats_prst)