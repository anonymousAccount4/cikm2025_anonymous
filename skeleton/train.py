import argparse
import os
import random

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from models.model import Model
from utils.data_utils import trans_list
from utils.argparser import init_parser, init_sub_args
from utils.dataset import get_dataset_and_loader
from tqdm import tqdm
import time


if __name__== '__main__':
    # Parse command line arguments and load config file
    parser = init_parser()
    args = parser.parse_args()
    args = init_sub_args(args)
    device = args.device

    # Set seeds    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    if args.use_wandb:
        tm = time.localtime()
        name = f'{tm.tm_mon}_{tm.tm_mday}_{tm.tm_hour}_{tm.tm_min}_{args.name}'
        args.exp_dir = f'{args.exp_dir}/{name}'
        callbacks = [LearningRateMonitor(logging_interval='step')]
        wandb_logger = WandbLogger(project=args.project_name, name=name, config=vars(args), log_model='all')
    else:
        wandb_logger = False
        callbacks = []
 
    # Set callbacks and logger
    if args.save:
        callbacks += [ModelCheckpoint(dirpath=args.exp_dir, monitor='AUC', mode='max')]

    # Get dataset and loaders
    # checkpoint_spatial = vars(args).get('checkpoint_spatial', None)
    # checkpoint_temporal = vars(args).get('checkpoint_temporal', None)
    checkpoint = vars(args).get('checkpoint', None)
    dataset, loader = get_dataset_and_loader(args, trans_list=trans_list, only_test=(checkpoint is not None))

    trainer = pl.Trainer(accelerator='gpu', default_root_dir=args.exp_dir, max_epochs=args.epochs, logger=wandb_logger, callbacks=callbacks, 
                        log_every_n_steps=20, num_sanity_val_steps=0, deterministic=True)

    if checkpoint is None:
        # Initialize model and trainer
        model = Model(args, dataset['test'].metadata, dataset['test'].V).to(device)
        
        # Train the model    
        trainer.fit(model=model, train_dataloaders=loader['train'], val_dataloaders=loader['test'])
    
    else:
        # Load model and trainer
        model = Model.load_from_checkpoint(checkpoint, args=args, metadata=dataset['test'].metadata, n_joints=dataset['test'].V).to(device)

        # Test the model
        trainer.test(model, dataloaders=loader['test'])
        