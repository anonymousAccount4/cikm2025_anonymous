import numpy as np
import torch, cv2
from utils.dataset import get_dataset_and_loader
from utils.argparser import init_parser, init_sub_args
from utils.data_utils import trans_list
import matplotlib.pyplot as plt

def vis_skeleton(keypoints, keypoint_preds, mean=None, std=None, im_res=(1280, 720)):
    '''
    keypoints: CxTxV
    keypoint_preds: CxTxV
    im_res: im_res of predictions

    return rendered image
    '''
    
    w, h = im_res
    keypoints = keypoints.transpose(1,2,0)
    keypoint_preds = keypoint_preds.transpose(1,2,0)
    
    if std is not None:
        keypoints = keypoints * std[None, None, None]
        keypoint_preds = keypoint_preds * std[None, None, None]
    
    if mean is not None:
        keypoints = keypoints + mean[None, None, :]
        keypoint_preds = keypoint_preds + mean[None, None, :]

    len_key = keypoints.shape[-2]
    if len_key == 18:
        # openpose(18)
        l_pair = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
                (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
    elif len_key == 17:
        # alphapose(17)
        l_pair  = [(0, 1), (0, 2), (1, 3), (2, 4), 
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                (3, 5), (4, 6), (5, 11), (6, 12), (11, 12),
                (11, 13), (12, 14), (13, 15), (14, 16)]
        
    return_images = []
    for i in range(len(keypoints)):
        ret = np.full((h, w, 3), 255, np.uint8)
        kp = keypoints[i]
        kp_pred = keypoint_preds[i]
        part_line = {}
        part_line_pred = {}
        # Draw keypoints
        for n in range(kp_pred.shape[0]):
            cor_x, cor_y = int((kp[n, 0]+1) / 2 * w), int((kp[n, 1]+1)/2 * h)
            p_cor_x, p_cor_y = int((kp_pred[n, 0]+1)/2 * w), int((kp_pred[n, 1]+1)/2 * h)
            part_line[n] = (cor_x, cor_y)
            part_line_pred[n] = (p_cor_x, p_cor_y)
            cv2.circle(ret, (cor_x, cor_y), 1, (0,0,0), 2)
            cv2.circle(ret, (p_cor_x, p_cor_y), 1, (204,0,0), 2)
            cv2.putText(ret, f'{n}', (cor_x, cor_y), 0, 1, (102, 255, 102))

        # Draw limbs
        for k, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]

                p_start_xy = part_line_pred[start_p]
                p_end_xy = part_line_pred[end_p]
                # if i < len(line_color):
                #     if opt.tracking:
                #         cv2.line(img, start_xy, end_xy, color, 2 * int(kp_scores[start_p] + kp_scores[end_p]) + 1)
                #     else:
                #         cv2.line(img, start_xy, end_xy, line_color[i], 2 * int(kp_scores[start_p] + kp_scores[end_p]) + 1)
                # cv2.line(ret, start_xy, end_xy, (102,102,102), 1)
                cv2.line(ret, p_start_xy, p_end_xy, (153, 0, 0), 1)

        return_images.append(ret)
        # cv2.imwrite(f'./results/{i}.png', ret)

    return return_images


def plot_result(gt, pred, key, save_path):
    x = np.arange(len(gt))
    plt.figure(figsize=(10,5))
    plt.ylim([0, 1])
    anomaly_idx = np.where(gt==1)[0]
    plt.bar(anomaly_idx, 1, width=1, color='r', alpha=0.5, label='Ground-truth')
    plt.plot(x, pred, 'b')
    plt.xlabel('Frame number', fontsize=16)
    plt.ylabel('Anomaly Score', fontsize=16)
    plt.legend(fontsize=16)
    plt.savefig(f'{save_path}/{key}.png')
    plt.close()