import argparse
from model import AutoEncoder
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import Feature_Track_Dataset
from train import train_AE
from utils import init_seed
from tqdm import tqdm

def main(args):
    init_seed(66)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = AutoEncoder(in_channel=args.feature).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99, last_epoch=-1, verbose=False)

    train_dataset = Feature_Track_Dataset(f'./features/{args.dataset}/train/{args.model}_loc_{args.t}_feat.h5',
                                            dataset=args.dataset, test=False)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, drop_last=True)

    test_dataset = Feature_Track_Dataset(f'./features/{args.dataset}/test/{args.model}_loc_{args.t}_feat.h5',
                                            dataset=args.dataset, test=True)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1)
    auc = train_AE(model, args.epoch, train_loader, test_loader, optimizer, test_dataset.labels, f'{args.dataset}_{args.model}', device, t=args.t)
    print(auc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--t', type=int, default=8)
    parser.add_argument('--dataset', type=str, default='shanghai',
                        choices=['shanghai', 'nwpu', 'ubnormal', 'avenue'], help='Dataset for Eval')
    parser.add_argument('--epoch', required=False, help='training epoch', type=int, default=50)
    parser.add_argument('--model', type=str, default='vit_b_16',
                        choices=['vit_b_16', 'vit_l_14', 'vit_b_32', 'rn50', 'rn101'], help='Model for Feature Extraction')
    parser.add_argument('--feature', type=int, default=512, help='Feature Dimension')
    parser.add_argument('--lr', required=False, type=float, default=3e-4)
    parser.add_argument('--batch_size', required=False, type=int, default=16)
    parser.add_argument('--seed', required=False, type=int, default=42)

    args = parser.parse_args()
    print(args)
    main(args)