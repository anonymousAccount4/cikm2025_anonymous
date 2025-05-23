import argparse
from model import AutoEncoder
import torch
from torch.utils.data import DataLoader
from dataset import Feature_Track_Dataset
import numpy as np
import torch.backends.cudnn as cudnn
import random 
import torch.nn.functional as F
import torch.nn as nn
import os, math
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import time
from utils import MinMaxNorm, smooth_scores

def init_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)

def plot_result(gt, pred, key, sf, ef):
    x = np.arange(ef-sf)
    plt.figure(figsize=(20,10))
    plt.ylim([0, 1.0])
    plt.fill_between(x, gt[sf:ef], alpha=0.2, color='r')
    plt.plot(x, pred[sf:ef], 'b', label='pred')
    plt.savefig(f'./results/plot_glo_0/{key}.png')
    plt.close()

def plot_roc(fpr, tpr):
    # plot model roc curve
    plt.plot(fpr, tpr, marker='.', label='Logistic')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    plt.savefig('curve.png')
    plt.close()

def preprocessing(x):
    org_l = len(x)
    new_l = math.ceil(org_l/4)*4
    x = F.pad(x.unsqueeze(0), (0, 0, 0, new_l - org_l), 'replicate').squeeze(0)
    return x

def eval(model, test_loader, labels, device='cpu', t=8):
    pred_dict = {}
    for pst, prst, ftr, video, chunk, info in test_loader:
        label = labels[video[0]]
        feature = pst.shape[-1]
        pst = pst.to(device).reshape(-1, feature)
        prst = prst.to(device).reshape(-1, feature)
        ftr = ftr.to(device).reshape(-1, feature)
        prst_ = preprocessing(prst)
        hat_pst, hat_prst, hat_ftr = model(prst_)
        hat_pst = hat_pst[:len(pst)]
        hat_prst = hat_prst[:len(prst)]
        hat_ftr = hat_ftr[:len(ftr)]
        output = torch.sum(torch.stack([torch.mean((prst-hat_prst)**2, 1), torch.mean((pst-hat_pst)**2, 1), torch.mean((ftr-hat_ftr)**2, 1)]), dim=0)
        tracks = torch.split(output, chunk)
        max_pred = torch.zeros(len(label))
        for i in range(len(tracks)):
            pred = F.interpolate(tracks[i].reshape(1,1,-1), scale_factor=t, mode='nearest').reshape(-1)
            start, end = info[i]
            pred_ = torch.zeros(len(label))
            if end-start+1 > len(pred):
                pred = F.pad(pred.reshape(1,1,-1), (0, (end-start+1)-len(pred)), 'replicate').reshape(-1)
            elif end-start+1 < len(pred):
                continue
            pred_[start:end+1] = pred
            max_pred = torch.max(torch.stack([max_pred, pred_]),dim=0)[0]

        pred_dict[video[0]] = max_pred.detach().cpu().numpy()
    
    pred_all = []
    gt_all = []
    for video in sorted(labels.keys()):
        if video not in pred_dict:
            pred_all.append(torch.zeros(labels[video].shape))
        else:
            pred_all.append(pred_dict[video])
        gt_all.append(labels[video])

    pred_all = MinMaxNorm(smooth_scores(pred_all))
    pred_all, gt_all = np.concatenate(pred_all, axis=0), np.concatenate(gt_all, axis=0)
    score = roc_auc_score(gt_all, pred_all)
    return score, pred_all

def main(args):
    init_seed(66)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = AutoEncoder(in_channel=args.feature).to(device)
    model.load_state_dict(torch.load(f'./results/models/{args.dataset}_{args.model}.pth'))
    model.eval()
    test_dataset = Feature_Track_Dataset(f'./features/{args.dataset}/test/{args.model}_loc_{args.t}_feat.h5',
                                            dataset=args.dataset, test=True)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1)
    with torch.no_grad():
        auc, score = eval(model, test_loader, test_dataset.labels, device=device, t=args.t)
    print(score.shape, score.min(), score.max())
    np.save(f'../final_score/{args.dataset}/{args.model}_loc_score.npy', score)
    print(f'dataset: {args.dataset}, model: {args.model}, auc: {auc}')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--t', type=int, default=8)
    parser.add_argument('--dataset', type=str, default='shanghai',
                        choices=['shanghai', 'nwpu', 'ubnormal', 'avenue'], help='Dataset for Eval')
    parser.add_argument('--model', type=str, default='vit_b_16',
                        choices=['vit_b_16', 'vit_l_14', 'vit_b_32', 'rn50', 'rn101'], help='Model for Feature Extraction')
    parser.add_argument('--feature', type=int, default=512, help='Feature Dimension')
    args = parser.parse_args()
    main(args)

