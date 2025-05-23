import os
import numpy as np
from sklearn.metrics import roc_auc_score

def calc_macro_auc(gt, pred):
    macro_auc = roc_auc_score(
        np.concatenate(([0], gt, [1])), np.concatenate(([0], pred, [1]))
    )
    return macro_auc


if __name__ == "__main__":
    dataset = "nwpu"
    path = f"./labels/{dataset}"
    labels = sorted(os.listdir(path))
    gt_all = []
    video_split = {}
    sf = 0
    for lab in labels:
        vid = lab.split(".")[0]
        label = np.load(os.path.join(path, lab))
        video_split[vid] = (sf, sf + len(label))
        sf += len(label)
        gt_all.append(label)
    gt_all = np.concatenate(gt_all, axis=0)
    
    skl_score, loc_score, glo_score = (
        np.load(f"./final_score/{dataset}/skl_score.npy"),
        np.load(f"./final_score/{dataset}/loc_score.npy"),
        np.load(f"./final_score/{dataset}/glo_score.npy"),
    )

    if dataset == "ubnormal":
        alpha, betta, gamma = 1.0, 0.1, 0.1
    elif dataset == "shanghai":
        alpha, betta, gamma = 1.0, 0.1, 0.1
    elif dataset == "nwpu":
        alpha, betta, gamma = 1.0, 0.1, 0.1

    score = alpha*skl_score + betta*loc_score + gamma*glo_score
    score = (score - score.min()) / (score.max() - score.min())
    micro_auc = roc_auc_score(gt_all, score)
    print(f'micro-auc: {micro_auc}')

    macro_auc = 0.0
    for key, (sf, ef) in video_split.items():
        macro_auc += calc_macro_auc(gt_all[sf:ef], score[sf:ef])
    print(f"macro-auc: {macro_auc / len(video_split)}")
