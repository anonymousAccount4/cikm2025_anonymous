import os
import pickle
import time
import argparse


def init_args():
    parser = init_parser()
    args = parser.parse_args()
    return init_sub_args(args)

def list_of_ints(arg):
    return list(map(int, arg.split(',')))

def init_sub_args(args):
    dataset = args.dataset
    
    args.pose_path = {'train': os.path.join(args.data_dir, dataset, 'pose', 'train/'),
                        'test':  os.path.join(args.data_dir, dataset, 'pose', 'test/')}

    args.ckpt_dir = None
    return args

def init_parser(default_data_dir='data/', default_exp_dir='data/exp_dir'):
    parser = argparse.ArgumentParser(prog="skpaint")
    # General Args
    parser.add_argument('--dataset', type=str, default='ubnormal',
                        choices=['shanghai', 'nwpu', 'ubnormal'], help='Dataset for Eval')
    parser.add_argument('--vid_res', type=list_of_ints, default=[856, 480], help='Video Res')
    parser.add_argument('--device', type=str, default='cuda:0', metavar='DEV', help='Device for feature calculation (default: \'cuda:0\')')
    parser.add_argument('--seed', type=int, metavar='S', default=999, help='Random seed, use 999 for random (default: 999)')
    # parser.add_argument('--verbose', type=int, default=1, metavar='V', choices=[0, 1], help='Verbosity [1/0] (default: 1)')
    parser.add_argument('--data_dir', type=str, default=default_data_dir, metavar='DATA_DIR', help="Path to directory holding .npy and .pkl files (default: {})".format(default_data_dir))
    parser.add_argument('--exp_dir', type=str, default=default_exp_dir, metavar='EXP_DIR', help="Path to the directory where models will be saved (default: {})".format(default_exp_dir))
    parser.add_argument('--num_workers', type=int, default=8, metavar='W', help='number of dataloader workers (0=current thread) (default: 32)')
    parser.add_argument('--plot_vis', action='store_true', help='Plot test skeleton') 

    # WANDB configuration
    parser.add_argument('--use_wandb', action='store_true', help='use wandb')
    parser.add_argument('--project_name', type=str, default="project_name")
    parser.add_argument('--save', action='store_true', help='Select whether to save the output')
    parser.add_argument('--name', type=str, default="name of experiment in wandb")
    parser.add_argument('--len_plot', type=int, default=4, help='number of plot segments')

    # Data Params
    parser.add_argument('--num_transform', type=int, default=2, metavar='T', help='number of transformations to use for augmentation (default: 2)')
    parser.add_argument('--num_coords', type=int, default=2, help='number of coordinates 2d or 3d to use for training (default: 2)')
    parser.add_argument('--headless', action='store_true', help='Remove head keypoints (14-17) and use 14 kps only. (default: False)')
    parser.add_argument('--ip_method', default='zero', choices=['zero', 'gaussian'], help='The choice method for inpainting points (zero or gaussian)')
    parser.add_argument('--ip_conf', type=float, default=1.0, help='How much noise should be injected into the points')
    parser.add_argument('--symm_range', action='store_true', help='Means shift data to [-1, 1] range')
    parser.add_argument('--kp18_format', action='store_true', help='joint translates 17 to 18')
    parser.add_argument('--train_seg_conf_th', '-th', type=float, default=0.0, metavar='CONF_TH', help='Training set threshold Parameter (default: 0.0)')
    parser.add_argument('--seg_len', type=int, default=16, metavar='SGLEN', help='Number of frames for training segment sliding window, a multiply of 6 (default: 12)')
    parser.add_argument('--seg_stride', type=int, default=4, metavar='SGST', help='Stride for training segment sliding window')
    parser.add_argument('--specific_clip', type=int, default=None, help='Train and Eval on Specific Clip')
    parser.add_argument('--global_pose_segs', action='store_false', help='Use unormalized pose segs')

    # Model Params
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--checkpoint', type=str, metavar='model', help="Path to a pretrained model")
    parser.add_argument('--checkpoint_spatial', type=str, metavar='model', help="Path to a pretrained model")
    parser.add_argument('--checkpoint_temporal', type=str, metavar='model', help="Path to a pretrained model")
    parser.add_argument('--batch_size', type=int, default=256,  metavar='B', help='Batch size for train')
    parser.add_argument('--n_cnn_layers', type=int, default=2, help='number of cnn layers')
    parser.add_argument('--dropout', type=float, default=0.0, help='probability of layer to drop')
    parser.add_argument('--epochs', '-model_e', type=int, default=100, metavar='E', help = 'Number of epochs per cycle')
    parser.add_argument('--loss_fn', type=str, default='mse', choices=['l1', 'smooth_l1', 'mse'], help='loss function')
    parser.add_argument('--mask_type', type=str, default='spatial', choices=['spatial', 'temporal', 'spatial_temporal'], help='mask type')
    # parser.add_argument('--model_optimizer', '-model_o', type=str, default='adamx', metavar='model_OPT', help = "Optimizer")
    # parser.add_argument('--model_sched', '-model_s', type=str, default='exp_decay', metavar='model_SCH', help = "Optimization LR scheduler")
    parser.add_argument('--model_lr', type=float, default=1e-3, metavar='LR', help='Optimizer Learning Rate Parameter')
    parser.add_argument('--layer_channels', type=list_of_ints, default='64,64,128,128')
    # parser.add_argument('--model_weight_decay', '-model_wd', type=float, default=5e-5, metavar='WD', help='Optimizer Weight Decay Parameter')
    # parser.add_argument('--model_lr_decay', '-model_ld', type=float, default=0.99, metavar='LD', help='Optimizer Learning Rate Decay Parameter')
    parser.add_argument('--model_hidden_dim', type=int, default=32, help='Features dim dimension')
    parser.add_argument('--model_latent_dim', type=int, default=64, help='Features dim dimension')
    parser.add_argument('--edge_importance', action='store_true', help='Adjacency matrix edge weights')
    parser.add_argument('--strategy', type=str, default='uniform', help='Adjacency matrix strategy')
    parser.add_argument('--max_hops', type=int, default=8, help='Adjacency matrix neighbours')
    return parser