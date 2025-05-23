import argparse

import torch
import torch.optim as optim
from dataset import Feature_Dataset
from model import AutoEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm
from train import train_AE
from utils import init_seed


def main(args):
    init_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoEncoder(in_features=512).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    train_dataset = Feature_Dataset(
        f"./features/{args.dataset}/train/glo_{args.time}_{args.src}_feat.h5",
        f"./features/{args.dataset}/train/glo_{args.time}_{args.tar}_feat.h5",
        args.dataset,
        False
    )
    train_loader = DataLoader(
        train_dataset, shuffle=True, batch_size=args.batch_size
    )

    test_dataset = Feature_Dataset(
        f"./features/{args.dataset}/test/glo_{args.time}_{args.src}_feat.h5",
        f"./features/{args.dataset}/test/glo_{args.time}_{args.tar}_feat.h5",
        args.dataset,
        True
    )
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1)
    auc = train_AE(
        model,
        args.epoch,
        train_loader,
        test_loader,
        optimizer,
        f"{args.dataset}_{args.src}",
        device,
        factor=args.time,
    )
    print(auc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--time",
        type=int,
        default=8
    )
    parser.add_argument(
        "--epoch", required=False, help="training epoch", type=int, default=50
    )
    parser.add_argument(
        "--src",
        type=str,
        default="Rec"
    )
    parser.add_argument(
        "--tar",
        type=str,
        default="Rec"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="shanghai",
        choices=["shanghai", "nwpu", "ubnormal"],
        help="Dataset for Eval",
    )
    parser.add_argument("--lr", required=False, type=float, default=3e-4)
    parser.add_argument("--batch_size", required=False, type=int, default=16)
    parser.add_argument("--seed", required=False, type=int, default=66)

    args = parser.parse_args()
    print(args)
    main(args)
