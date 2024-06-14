"""
Copyright (C) 2024  ETH Zurich, Hsuan-I Ho
"""
import os, sys
from datetime import datetime
import logging as log
import numpy as np
import torch
import random
import shutil
import tempfile
import wandb

from torch.utils.data import DataLoader
from datasets.train_dataset import TrainReconDataset
from models.nrm_trainer import NrmTrainer
from utils.config import *


def create_archive(save_dir, config):

    with tempfile.TemporaryDirectory() as tmpdir:

        shutil.copy(config, os.path.join(tmpdir, 'config.yaml'))
        shutil.copy('train_normal.py', os.path.join(tmpdir, 'train_normal.py'))

        for name in ['models', 'datasets', 'utils']:
            shutil.copytree(
                os.path.join(name),
                os.path.join(tmpdir, name),
                ignore=shutil.ignore_patterns('__pycache__'))

        shutil.make_archive(
            os.path.join(save_dir, 'code_copy'),
            'zip',
            tmpdir) 


def main(config):

    # Set random seed.
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    log_dir = os.path.join(
            config.save_root,
            config.exp_name,
            f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        )

    # Backup code.
    create_archive(log_dir, config.config)
    
    # Initialize dataset and dataloader.

    dataset = TrainReconDataset(config.data_root, config)

    loader = DataLoader(dataset=dataset, 
                        batch_size=config.batch_size, 
                        shuffle=True, 
                        num_workers=config.workers,
                        pin_memory=True)
    
    trainer = NrmTrainer(config, log_dir)


    if config.wandb_id is not None:
        wandb_id = config.wandb_id
    else:
        wandb_id = wandb.util.generate_id()
        with open(os.path.join(log_dir, 'wandb_id.txt'), 'w+') as f:
            f.write(wandb_id)

    wandb_mode = "disabled" if (not config.wandb) else "online"
    wandb.init(id=wandb_id,
               project=config.wandb_name,
               config=config,
               name=os.path.basename(log_dir),
               resume="allow",
               settings=wandb.Settings(start_method="fork"),
               mode=wandb_mode,
               dir=log_dir)
    wandb.watch(trainer)

    if config.resume:
        trainer.load_checkpoint(config.resume)

    global_step = trainer.global_step
    start_epoch = trainer.epoch


    for epoch in range(start_epoch, config.epochs):
        for data in loader:
            trainer.step(epoch=epoch, n_iter=global_step, data=data)
            
            if global_step % config.log_every == 0:
                trainer.log(global_step, epoch)

            global_step += 1

        if epoch % config.save_every == 0:
            trainer.save_checkpoint(full=False)

    wandb.finish()

if __name__ == "__main__":

    parser = parse_options()
    args, args_str = argparse_to_str(parser)
    handlers = [log.StreamHandler(sys.stdout)]
    log.basicConfig(level=args.log_level,
                        format='%(asctime)s|%(levelname)8s| %(message)s',
                        handlers=handlers)
    log.info(f'Info: \n{args_str}')
    main(args)