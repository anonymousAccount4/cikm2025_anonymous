import argparse
import math
import os
import random

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from dataset import Feature_Dataset
from model import AutoEncoder
from sklearn.metrics import auc, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from utils import MinMaxNorm, smooth_scores
import time

feature_dict = {
    'vit_b_16': 512,
    'vit_l_14': 768,
    'vit_b_32': 512,
    'rn50': 1024,
    'rn101': 512
}

def init_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)


def plot_result(gt, pred, key, sf, ef):
    x = np.arange(ef - sf)
    plt.figure(figsize=(20, 10))
    plt.ylim([0, 1.0])
    plt.fill_between(x, gt[sf:ef], alpha=0.2, color="r")
    plt.plot(x, pred[sf:ef], "b", label="pred")
    plt.savefig(f"./results/plot_glo_0/{key}.png")
    plt.close()


def plot_roc(fpr, tpr):
    # plot model roc curve
    plt.plot(fpr, tpr, marker=".", label="Logistic")
    # axis labels
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    # show the legend
    plt.legend()
    plt.savefig("curve.png")
    plt.close()


def preprocessing(x):
    org_l = len(x)
    new_l = math.ceil(org_l / 8) * 8
    x = F.pad(x.unsqueeze(0), (0, 0, 0, new_l - org_l), "replicate").squeeze(0)
    return x

def eval(model, test_loader, device="cpu", factor=8):
    gt_all, pred_all = [], []
    for pst, prst, ftr, label in test_loader:
        label = label.squeeze(0)
        pst = pst.to(device).squeeze(0)
        prst = prst.to(device).squeeze(0)
        ftr = ftr.to(device).squeeze(0)
        hat_pst, hat_prst, hat_ftr = model(prst)
        pred = torch.mean((pst - hat_pst) ** 2, [-1]) + torch.mean((prst - hat_prst) ** 2, [-1]) + torch.mean((ftr - hat_ftr) ** 2, [-1])

        pred = F.interpolate(pred.unsqueeze(0).unsqueeze(0), scale_factor=factor, mode="nearest").squeeze()
        pred = F.pad(pred, (0, len(label) - len(pred)), "constant", pred[-1]).cpu().numpy()

        gt_all.append(label)
        pred_all.append(pred)
    pred_all = MinMaxNorm(smooth_scores(pred_all))

    pred_all, gt_all = (
        np.concatenate(pred_all, axis=0),
        np.concatenate(gt_all, axis=0),
    )

    score = roc_auc_score(gt_all, pred_all)
    return score, pred_all


def main(args):
    init_seed(66)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_size = feature_dict[args.model]
    model = AutoEncoder(in_channel=feature_size).to(device)
    model.load_state_dict(torch.load(f"./results/{args.dataset}_{args.model}.pth"))
    model.eval()

    test_dataset = Feature_Dataset(
        f"./features/{args.dataset}/test/{args.model}_glo_8_feat.h5",
        args.dataset,
        True
    )
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1)
    with torch.no_grad():
        auc, score = eval(
            model,
            test_loader,
            device=device,
            factor=args.time
        )
    print(score.shape, score.min(), score.max())
    np.save(f"../final_score/{args.dataset}/{args.model}_glo_score.npy", score)
    print(auc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--time",
        type=int,
        default="8"
    )
    parser.add_argument(
        "--epoch", required=False, help="training epoch", type=int, default=50
    )
    parser.add_argument('--model', type=str, default='vit_b_16',
                        choices=['vit_b_16', 'vit_l_14', 'vit_b_32', 'rn50', 'rn101'], help='Model for Feature Extraction')
    parser.add_argument(
        "--dataset",
        type=str,
        default="shanghai",
        choices=["shanghai", "nwpu", "ubnormal"],
        help="Dataset for Eval",
    )
    args = parser.parse_args()
    main(args)
