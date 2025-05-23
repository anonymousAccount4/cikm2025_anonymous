import os
import re
import numpy as np
from scipy.ndimage import gaussian_filter1d
from utils.dataset import shanghaitech_hr_skip

def smooth_scores(scores_arr, sigma=9):
    scores_arr_ = scores_arr.copy()
    for s in range(len(scores_arr_)):
        for sig in range(1, sigma):
            scores_arr_[s] = gaussian_filter1d(scores_arr_[s], sigma=sig)
    return scores_arr_


def get_clip_score(scores, clip, metadata_np, metadata, per_frame_scores_root, args):
    if args.dataset == 'ubnormal':
        type, scene_id, clip_id = re.findall('(abnormal|normal)_scene_(\d+)_scenario(.*).npy', clip)[0]
        clip_id = type + "_" + clip_id
    elif args.dataset == 'nwpu':
        scene_id, clip_id = clip.split('.')[0].split('_')
        scene_id = int(scene_id[1:])
        clip_id = int(clip_id)
    elif args.dataset == 'shanghai':
        scene_id, clip_id = [int(i) for i in clip.replace("label", "001").split('.')[0].split('_')]
    clip_metadata_inds = np.where((metadata_np[:, 1] == clip_id) &
                                  (metadata_np[:, 0] == scene_id))[0]
    clip_metadata = metadata[clip_metadata_inds]
    clip_fig_idxs = set([arr[2] for arr in clip_metadata])
    clip_res_fn = os.path.join(per_frame_scores_root, clip)
    clip_gt = np.load(clip_res_fn)

    scores_zeros = np.zeros(clip_gt.shape[0])
    if args.dataset == "ubnormal":
        scores_zeros = np.zeros(clip_gt.shape[0]+1)
    if len(clip_fig_idxs) == 0:
        clip_person_scores_dict = {0: np.copy(scores_zeros)}
    else:
        clip_person_scores_dict = {i: np.copy(scores_zeros) for i in clip_fig_idxs}

    for person_id in clip_fig_idxs:
        person_metadata_inds = \
            np.where(
                (metadata_np[:, 1] == clip_id) & (metadata_np[:, 0] == scene_id) & (metadata_np[:, 2] == person_id))[0]
        pid_scores = scores[person_metadata_inds]

        pid_frame_inds = np.array([metadata[i][3] for i in person_metadata_inds]).astype(int)
        clip_person_scores_dict[person_id][pid_frame_inds + int(args.seg_len)] = pid_scores

        if scene_id == 1 and clip_id == 130:
            tmp = pid_frame_inds + int(args.seg_len)
            if 130 in tmp:
                print(person_id, pid_scores[tmp==130])
            elif 160 in tmp:
                print(person_id, pid_scores[tmp==160])

    clip_ppl_score_arr = np.stack(list(clip_person_scores_dict.values()))
    clip_score = np.amax(clip_ppl_score_arr, axis=0)
    if args.dataset == 'ubnormal':
        clip_score = clip_score[:-1]
    
    return clip_gt, clip_score

def get_dataset_scores(scores, metadata, args):
    dataset_gt_arr = []
    dataset_scores_arr = []
    metadata_np = np.array(metadata)
    
    per_frame_scores_root = f'../labels/{args.dataset}'
    clip_list = os.listdir(per_frame_scores_root)
    clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy'))

    for clip in clip_list:
        clip_gt, clip_score = get_clip_score(scores, clip, metadata_np, metadata, per_frame_scores_root, args)
        if clip_score is not None:
            dataset_gt_arr.append(clip_gt)
            dataset_scores_arr.append(clip_score)
    
    return dataset_gt_arr, dataset_scores_arr, clip_list

def MinMaxNorm(x):
    x = np.concatenate(x)
    return (x - x.min())/(x.max()-x.min())