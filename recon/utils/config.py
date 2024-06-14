"""
Copyright (C) 2024  ETH Zurich, Hsuan-I Ho
"""
import argparse
import pprint
import yaml
import logging

def parse_options():

    parser = argparse.ArgumentParser(description='SiTH Code')


    ###################
    # Global arguments
    ###################
    global_group = parser.add_argument_group('global')
    global_group.add_argument('--config', type=str, default='config.yaml', 
                               help='Path to config file to replace defaults')
    global_group.add_argument('--save_root', type=str, default='./checkpoints/', 
                               help="outputs path")
    global_group.add_argument('--exp_name', type=str, default='test',
                               help="Experiment name.")
    global_group.add_argument('--seed', type=int, default=2434)
    global_group.add_argument('--resume', type=str, default=None,
                                help='Resume from the checkpoint path.')
    global_group.add_argument(
        '--log_level', action='store', type=int, default=logging.INFO,
        help='Logging level to use globally, DEBUG: 10, INFO: 20, WARN: 30, ERROR: 40.')
        
    ###################
    # Arguments for dataset
    ###################
    data_group = parser.add_argument_group('dataset')
    data_group.add_argument('--data_root', type=str, default='',
                            help='Path to dataset')
    data_group.add_argument('--num_samples', type=int, default=20480,
                            help='Number of samples to use for each subject during training')
    data_group.add_argument('--img_size', type=int, default=1024,
                            help='Image size for training')
    data_group.add_argument("--white_bg", action="store_true", default=False,
        help="Whether to use a white background for the training images. If False, a random background will be added.")
    data_group.add_argument("--aug_jitter", action="store_true", default=False,
                            help="Whetherto use data augmentation with jittering.")   

    ###################
    # Arguments for optimizer
    ###################
    optim_group = parser.add_argument_group('optimizer')
    optim_group.add_argument('--lr_decoder', type=float, default=0.001,
                             help='Learning rate for the decoder.')
    optim_group.add_argument('--lr_encoder', type=float, default=0.0001,
                             help='Learning rate for the encoder.')
    optim_group.add_argument('--beta1', type=float, default=0.5,
                                help='Beta1.')
    optim_group.add_argument('--beta2', type=float, default=0.999,
                                help='Beta2.')
    optim_group.add_argument('--weight_decay', type=float, default=0.0, 
                             help='Weight decay.')

    ###################
    # Arguments for scheduler
    ###################

    scheduler_group = parser.add_argument_group('scheduler')

    scheduler_group.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    scheduler_group.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    scheduler_group.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    scheduler_group.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    scheduler_group.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    ###################
    # Arguments for training
    ###################
    train_group = parser.add_argument_group('train')
    train_group.add_argument('--epochs', type=int, default=5000, 
                             help='Number of epochs to run the training.')
    train_group.add_argument('--batch_size', type=int, default=8, 
                             help='Batch size for the training.')
    train_group.add_argument('--workers', type=int, default=0,
                             help='Number of workers for the data loader. 0 means single process.')
    train_group.add_argument('--save_every', type=int, default=50, 
                             help='Save the model at every N epoch.')
    train_group.add_argument('--log_every', type=int, default=100,
                             help='write logs to wandb at every N iters')
    
    ###################
    # Arguments for Network
    ###################
    net_group = parser.add_argument_group('network')
    net_group.add_argument('--pos_dim', type=int, default=8,
                          help='input position dimension')
    net_group.add_argument('--feat_dim', type=int, default=16,
                          help='image endoer feature dimension')
    net_group.add_argument('--num_layers', type=int, default=8, 
                             help='Number of layers for the MLPs.')
    net_group.add_argument('--hidden_dim', type=int, default=256,
                          help='Network width')
    net_group.add_argument('--activation', type=str, default='relu',
                            choices=['relu', 'sin', 'softplus', 'lrelu'])
    net_group.add_argument('--layer_type', type=str, default='none',
                            choices=['none', 'spectral_norm', 'frobenius_norm', 'l_1_norm', 'l_inf_norm'])
    net_group.add_argument('--skip', type=int, nargs='*', default=[2],
                          help='Layer to have skip connection.')

    ###################
    # Embedder arguments
    ###################
    embedder_group = parser.add_argument_group('embedder')
    embedder_group.add_argument('--shape_freq', type=int, default=0,
                                help='log2 of max freq')
    embedder_group.add_argument('--color_freq', type=int, default=0,
                                help='log2 of max freq')

    ###################
    # Losses arguments
    ###################
    losses_group = parser.add_argument_group('losses')
    losses_group.add_argument('--lambda_sdf', type=float, default=10,
                                help='lambda for sdf loss')
    losses_group.add_argument('--lambda_rgb', type=float, default=10,
                                help='lambda for rgb loss')
    losses_group.add_argument('--lambda_nrm', type=float, default=1,
                                help='lambda for normal loss')
    losses_group.add_argument('--lambda_2D', type=float, default=1,
                                help='lambda for 2D normal loss')
    losses_group.add_argument("--use_mask", action="store_true", default=False,
                            help="Whether to use mask for the 2D normal loss.")
    losses_group.add_argument("--use_pred_nrm", action="store_true", default=False,
                            help="Whether to use predicted normal for reconstruction.")

   ###################
    # Arguments for validation
    ###################
    valid_group = parser.add_argument_group('validation')
    valid_group.add_argument("--valid", action="store_true", default=False,
                            help="Whether to run validation.")
    valid_group.add_argument('--valid_folder', type=str, default='',
                             help='Path to the validation dataset')
    valid_group.add_argument('--valid_every', type=int, default=50,
                             help='Frequency of running validation.')
    valid_group.add_argument('--num_valid_samples', type=int, default=5,
                                help='Number of samples to validate.')
    valid_group.add_argument('--subdivide', type=bool, default=True, 
                            help='Subdivide the mesh before marching cubes')
    valid_group.add_argument('--grid_size', type=int, default=512, 
                            help='Grid size for marching cubes')
    valid_group.add_argument('--erode_iter', type=int, default=0,
                            help='Number of iterations for mask erosion. 0 means no erosion.')
 
    ###################
    # Arguments for wandb
    ###################
    wandb_group = parser.add_argument_group('wandb')
    
    wandb_group.add_argument('--wandb_id', type=str, default=None,
                             help='wandb id')
    wandb_group.add_argument('--wandb', action='store_true',
                             help='Use wandb')
    wandb_group.add_argument('--wandb_name', default='default', type=str,
                             help='wandb_name')

    return parser


def parse_yaml_config(config_path, parser):
    """Parses and sets the parser defaults with a yaml config file.

    Args:
        config_path : path to the yaml config file.
        parser : The parser for which the defaults will be set.
        parent : True if parsing the parent yaml. Should never be set to True by the user.
    """
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    list_of_valid_fields = []
    for group in parser._action_groups:
        group_dict = {list_of_valid_fields.append(a.dest) for a in group._group_actions}
    list_of_valid_fields = set(list_of_valid_fields)
    
    defaults_dict = {}

    # Loads child parent and overwrite the parent configs
    # The yaml files assumes the argument groups, which aren't actually nested.
    for key in config_dict:
        for field in config_dict[key]:
            if field not in list_of_valid_fields:
                raise ValueError(
                    f"ERROR: {field} is not a valid option. Check for typos in the config."
                )
            defaults_dict[field] = config_dict[key][field]


    parser.set_defaults(**defaults_dict)

def argparse_to_str(parser, args=None):
    """Convert parser to string representation for Tensorboard logging.

    Args:
        parser (argparse.parser): Parser object. Needed for the argument groups.
        args : The parsed arguments. Will compute from the parser if None.
    
    Returns:
        args    : The parsed arguments.
        arg_str : The string to be printed.
    """
    
    if args is None:
        args = parser.parse_args()

    if args.config is not None:
        parse_yaml_config(args.config, parser)

    args = parser.parse_args()

    args_dict = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        args_dict[group.title] = vars(argparse.Namespace(**group_dict))

    pp = pprint.PrettyPrinter(indent=2)
    args_str = pp.pformat(args_dict)
    args_str = f'```{args_str}```'

    return args, args_str