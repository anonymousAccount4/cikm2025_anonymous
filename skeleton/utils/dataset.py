import json
import math
import os
import re, cv2
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.data_utils import normalize_pose, normalize_pose_robust
from utils.pose_utils import gen_clip_seg_data_np, get_ab_labels
from torch.utils.data import DataLoader
import faiss

SHANGHAITECH_HR_SKIP = [(1, 130), (1, 135), (1, 136), (6, 144), (6, 145), (12, 152)]


class PoseSegDataset(Dataset):
    """
    Generates a dataset with two objects, a np array holding sliced pose sequences
    and an object array holding file name, person index and start time for each sliced seq


    If path_to_patches is provided uses pre-extracted patches. If lmdb_file or vid_dir are
    provided extracts patches from them, while hurting performance.
    """

    def __init__(self, path_to_json_dir, normalize_pose_segs=True, return_indices=False,
                 return_metadata=False, kp18_format=False, debug=False, evaluate=False, **dataset_args):
        super().__init__()
        self.args = dataset_args
        self.path_to_json = path_to_json_dir
        self.patches_db = None
        self.use_patches = False
        self.normalize_pose_segs = normalize_pose_segs
        self.headless = dataset_args.get('headless', False)
        self.ip_method = dataset_args.get('ip_method', 'zero')
        self.ip_conf = dataset_args.get('ip_conf', 1.0)
        self.num_coords = dataset_args.get('num_coords', 2)
        self.mask_type = dataset_args.get('mask_type', 'spatial')
        self.eval = evaluate
        self.debug = debug
        self.kp18_format = kp18_format
        num_clips = dataset_args.get('specific_clip', None)
        self.return_indices = return_indices
        self.return_metadata = return_metadata
        self.transform_list = dataset_args.get('trans_list', None)
        if self.transform_list is None:
            self.apply_transforms = False
            self.num_transform = 1
        else:
            self.apply_transforms = True
            self.num_transform = len(self.transform_list)
        self.train_seg_conf_th = dataset_args.get('train_seg_conf_th', 0.0)
        self.seg_len = dataset_args.get('seg_len', 12)
        self.seg_stride = dataset_args.get('seg_stride', 1)
        self.segs_data_np, self.splits, self.segs_meta = \
            gen_dataset(path_to_json_dir, num_clips=num_clips, ret_keys=True, kp18_format=self.kp18_format, **dataset_args)        

        # Convert person keys to ints
        # self.person_keys = {k: [int(i) for i in v] for k, v in self.person_keys.items()}
        self.num_samples, self.C, self.T, self.V = self.segs_data_np.shape
        self.metadata = self.segs_meta

    def __getitem__(self, index):
        # Select sample and augmentation. I.e. given 5 samples and 2 transformations,
        # sample 7 is data sample 7%5=2 and transform is 7//5=1
        if self.apply_transforms:
            sample_index = index % self.num_samples
            trans_index = math.floor(index / self.num_samples)
            data_numpy = np.array(self.segs_data_np[sample_index])
            data_transformed = self.transform_list[trans_index](data_numpy)

        else:
            sample_index = index
            data_transformed = np.array(self.segs_data_np[index])
            trans_index = 0  # No transformations


        data_transformed = data_transformed[:self.num_coords,:,:]

        if self.normalize_pose_segs:
            data_transformed, data_mean, data_std = normalize_pose(data_transformed.transpose((1, 2, 0))[None, ...], **self.args)
            data_transformed = data_transformed.transpose(2, 0, 1)
        # split = self.splits[sample_index]

        data_target_transformed = data_transformed.copy()
        
        return data_transformed.astype(np.float32), data_target_transformed.astype(np.float32)

    def __len__(self):
        return self.num_transform * self.num_samples

def get_dataset_and_loader(args, trans_list, only_test=False):
    loader_args = {'batch_size': args.batch_size, 'num_workers': args.num_workers, 'pin_memory': True}
    dataset_args = {'headless': args.headless, 'seg_len': args.seg_len, 'return_indices': True, 'return_metadata': True, 'kp18_format': args.kp18_format
                    , "dataset": args.dataset, "symm_range": args.symm_range, 'train_seg_conf_th': args.train_seg_conf_th, 'specific_clip': args.specific_clip
                    , 'num_coords': args.num_coords, 'vid_res': args.vid_res, 'ip_method': args.ip_method, 'ip_conf': args.ip_conf, 'mask_type': args.mask_type}
    dataset, loader = dict(), dict()
    splits = ['train', 'test']
    for split in splits:
        evaluate = split == 'test'
        normalize_pose_segs = args.global_pose_segs
        dataset_args['trans_list'] = trans_list[:args.num_transform] if split == 'train' else None
        dataset_args['seg_stride'] = args.seg_stride if split == 'train' else 1  # No strides for test set
        dataset[split] = PoseSegDataset(args.pose_path[split],
                                        normalize_pose_segs=normalize_pose_segs,
                                        evaluate=evaluate,
                                        **dataset_args)
        loader[split] = DataLoader(dataset[split], **loader_args, shuffle=(split == 'train'))
    # if only_test:
    #     loader['train'] = None
    return dataset, loader


def shanghaitech_hr_skip(shanghaitech_hr, scene_id, clip_id):
    if not shanghaitech_hr:
        return shanghaitech_hr
    if (int(scene_id), int(clip_id)) in SHANGHAITECH_HR_SKIP:
        return True
    return False

def num2bool(splits, nodes=18):
    l = len(splits)
    ret = np.array([[False]*18 for _ in range(l)])
    for i in range(l):
        ret[i, splits[i]] = True
    return ret

def split_kp(data, metadata):
    splits = np.array([[2,3,4],     # left arm
                       [5,6,7],     # right arm
                       [8,9,10],    # left leg
                       [11,12,13]]) # right leg
    splits = num2bool(splits)
    l, j = splits.shape
    splits = splits[None, ...].repeat(len(data), 0).reshape(-1, j)
    metadata = np.array(metadata).repeat(l, 0)
    data = data.repeat(l, 0)

    return data, splits, metadata

def gen_dataset(person_json_root, num_clips=None, kp18_format=True, ret_keys=False, **dataset_args):
    segs_data_np = []
    segs_ids = []
    segs_meta = []
    person_keys = dict()
    start_ofst = dataset_args.get('start_ofst', 0)
    seg_stride = dataset_args.get('seg_stride', 1)
    mask_type = dataset_args.get('mask_type', 'spatial')
    seg_len = dataset_args.get('seg_len', 24)
    headless = dataset_args.get('headless', False)
    seg_conf_th = dataset_args.get('train_seg_conf_th', 0.0)
    dataset = dataset_args.get('dataset', 'ShanghaiTech')

    dir_list = os.listdir(person_json_root)
    json_list = sorted([fn for fn in dir_list if fn.endswith('tracked_person.json')])
    if num_clips is not None:
        json_list = [json_list[num_clips]]  # For debugging purposes
        
    for person_dict_fn in tqdm(json_list):
        if dataset == "ubnormal":
            type, scene_id, clip_id = \
                re.findall('(abnormal|normal)_scene_(\d+)_scenario(.*)_alphapose_.*', person_dict_fn)[0]
            clip_id = type + "_" + clip_id
        elif dataset == "nwpu":
            scene_id, clip_id = person_dict_fn.split('_')[:2]
            scene_id = scene_id[1:]
        elif dataset == "shanghai":
            scene_id, clip_id = person_dict_fn.split('_')[:2]
        clip_json_path = os.path.join(person_json_root, person_dict_fn)
        with open(clip_json_path, 'r') as f:
            clip_dict = json.load(f)
        clip_segs_data_np, clip_segs_meta, clip_keys, clip_segs_ids = gen_clip_seg_data_np(
            clip_dict, start_ofst,
            seg_stride,
            seg_len,
            scene_id=scene_id,
            clip_id=clip_id,
            ret_keys=ret_keys,
            dataset=dataset)
        segs_data_np.append(clip_segs_data_np)
        segs_meta += clip_segs_meta
        segs_ids += clip_segs_ids
        person_keys = {**person_keys, **clip_keys}

    segs_data_np = np.concatenate(segs_data_np, axis=0)
    if kp18_format and segs_data_np.shape[-2] == 17:
        segs_data_np = keypoints17_to_coco18(segs_data_np)
        
    if headless:
        segs_data_np = segs_data_np[:, :, 5:]
    
    if mask_type == 'temporal':
        splits = None
    else:
        segs_data_np, splits, segs_meta = split_kp(segs_data_np, segs_meta)

    segs_data_np = np.transpose(segs_data_np, (0, 3, 1, 2)).astype(np.float32)

    return segs_data_np, splits, np.array(segs_meta)


def keypoints17_to_coco18(kps):
    """
    Convert a 17 keypoints coco format skeleton to an 18 keypoint one.
    New keypoint (neck) is the average of the shoulders, and points
    are also reordered.
    """
    kp_np = np.array(kps)
    neck_kp_vec = 0.5 * (kp_np[..., 5, :] + kp_np[..., 6, :])
    kp_np = np.concatenate([kp_np, neck_kp_vec[..., None, :]], axis=-2)
    opp_order = [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
    opp_order = np.array(opp_order, dtype=np.int32)
    kp_coco18 = kp_np[..., opp_order, :]
    return kp_coco18


def seg_conf_th_filter(segs_data_np, segs_meta, segs_score_np, seg_conf_th=2.0):
    # seg_len = segs_data_np.shape[2]
    # conf_vals = segs_data_np[:, 2]
    # sum_confs = conf_vals.sum(axis=(1, 2)) / seg_len
    sum_confs = segs_score_np.mean(axis=1)
    seg_data_filt = segs_data_np[sum_confs > seg_conf_th]
    seg_meta_filt = list(np.array(segs_meta)[sum_confs > seg_conf_th])
    segs_score_np = segs_score_np[sum_confs > seg_conf_th]

    return seg_data_filt, seg_meta_filt, segs_score_np