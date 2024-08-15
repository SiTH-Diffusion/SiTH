"""
Copyright (C) 2023  ETH Zurich, Hsuan-I Ho
"""

import os
import sys
import logging as log
import numpy as np
import torch
import time
import random
import pickle

from torch.utils.data import DataLoader
from recon.models.evaluator import Evaluator
from recon.utils.config import parse_options, argparse_to_str
from recon.datasets.test_dataset import TestFolderDataset
from recon.models.ops.mesh.load_obj import load_obj

####################################################
CANONICAL_TEMPLATE = 'data/smplx_canonical.obj'
WATERTIGHT_TEMPLATE = 'data/smplx_watertight.pkl'
####################################################

def main(config):

    # Set random seed.
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    with open(WATERTIGHT_TEMPLATE, 'rb') as f:
        watertight = pickle.load(f)

    dataset = TestFolderDataset(args.test_folder, config, watertight)

    loader = DataLoader(dataset=dataset, 
                        batch_size=1, 
                        shuffle=False, 
                        num_workers=0,
                        pin_memory=True)

    can_V, _ = load_obj(CANONICAL_TEMPLATE)

    evaluator = Evaluator(config, watertight, can_V)

    save_path = os.path.join(args.test_folder, 'meshes')
    os.makedirs(save_path, exist_ok=True)

    for i, data in enumerate(loader):
        with torch.no_grad():
            evaluator.test_reconstruction(data, save_path, subdivide=args.subdivide, save_uv=args.save_uv)

if __name__ == "__main__":

    parser = parse_options()

    parser.add_argument('--test_folder', type=str, required=True, help='saving directory')
    parser.add_argument('--save_uv', action='store_true', help='save texture meshes with uv')

    args, args_str = argparse_to_str(parser)
    handlers = [log.StreamHandler(sys.stdout)]
    log.basicConfig(level=args.log_level,
                        format='%(asctime)s|%(levelname)8s| %(message)s',
                        handlers=handlers)
    log.info(f'Info: \n{args_str}')
    main(args)
