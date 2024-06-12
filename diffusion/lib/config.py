"""
Copyright (C) 2024  ETH Zurich, Hsuan-I Ho
"""
import argparse
import pprint
import yaml

def parse_options():

    parser = argparse.ArgumentParser(description='Back Hallucination Model Training Options')

    ###################
    # Global arguments
    ###################
    global_group = parser.add_argument_group('global')

    global_group.add_argument(
        "--config",
        type=str,
        default='config.yaml',
        help='Path to config file to replace defaults')
    global_group.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='kxic/zero123-xl',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    global_group.add_argument(
        "--output_dir",
        type=str,
        default="../checkpoints",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    global_group.add_argument(
        "--exp_name",
        type=str,
        default="train-hallucination",
        help="Experiment name for logging. Will default to test",
    )
    global_group.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    ###################
    # Dataset arguments
    ###################
    dataset_group = parser.add_argument_group('dataset')

    dataset_group.add_argument(
        "--train_data_dir",
        type=str,
        default='../data/dataset.h5',
        help=(
            "path of h5f file containing the training dataset"
        ),
    )
    dataset_group.add_argument(
        "--test_data_dir",
        type=str,
        default=None,
        help=(
            "path of folder containing the rgb images for testing"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    dataset_group.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    dataset_group.add_argument(
        "--sample_random_views",
        action="store_true",
        default=False,
        help=(
            "Whether to sample a random view pair for the training images. If False, only the back view will be used."
        ),
    )

    dataset_group.add_argument(
        "--white_background",
        action="store_true",
        default=False,
        help=(
            "Whether to use a white background for the training images. If False, a random background will be added."
        ),
    )

    ###################
    # Arguments for training
    ###################
    training_group = parser.add_argument_group('training')

    training_group.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    training_group.add_argument("--num_train_epochs", type=int, default=100)

    training_group.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )

    training_group.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    training_group.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    training_group.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    training_group.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    training_group.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    ###################
    # Arguments for validation
    ###################
    validation_group = parser.add_argument_group('validation')

    validation_group.add_argument(
        "--validation",
        action='store_true',
        help=("wheter to run validation and log training images during training."
        ),
    )
    validation_group.add_argument(
        "--test",
        action='store_true',
        help=("wheter to test images during training."
        ),
    )
    validation_group.add_argument(
        "--num_gen_images",
        type=int,
        default=4,
        help="Number of images to be generated for val/test images",
    )
    validation_group.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of training images to be validated",
    )
    validation_group.add_argument(
        "--validation_steps",
        type=int,
        default=2500,
        help=(
            "Run validation every X steps. Validation consists of running the first X images in the train dataset."
        ),
    )
    validation_group.add_argument(
        "--test_steps",
        type=int,
        default=2500,
        help=(
            "Run test every X steps. Test consists of running the unseen images"
        ),
    )
    ###################
    # Arguments for networks
    ###################

    network_group = parser.add_argument_group('network')

    network_group.add_argument(
        "--drop_prob",
        type=float,
        default=0.1,
        help="dropout probability for the classifier free guidence"
    )
    network_group.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="classifier free guidence scale"
    )
    network_group.add_argument(
        "--conditioning_scale",
        type=float,
        default=1.0,
        help="Controlnet conditioning scale"
    )
    network_group.add_argument(
        "--conditioning_channels",
        type=int,
        default=4,
        help=(
            "Number of conditioning channels for controlnet. If set to 4, the controlnet will be conditioned on the"
            " UV image and the mask. If set to 8, the controlnet will be conditioned on additional"
            " view point information."
        ),
    )
    ###################
    # Arguments for optimizer
    ###################

    optimizer_group = parser.add_argument_group('optimizer')


    optimizer_group.add_argument(
        "--lr",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )


    optimizer_group.add_argument(
        "--lr_controlnet",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use for ControlNet.",
    )

    optimizer_group.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    optimizer_group.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    optimizer_group.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    optimizer_group.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    optimizer_group.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    optimizer_group.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    optimizer_group.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    optimizer_group.add_argument("--adam_weight_decay", type=float, default=1e-5, help="Weight decay to use.")
    optimizer_group.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    optimizer_group.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    ###################
    # Arguments for optimizer
    ###################

    misc_group = parser.add_argument_group('misc')


    misc_group.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    misc_group.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
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