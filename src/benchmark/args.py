from transformers import SchedulerType
from argparse import ArgumentParser

import configs


def add_args(parser: ArgumentParser):

    # model
    parser.add_argument("--init_model", type=str, default="codet5",
                        choices=configs.model_to_ids.keys(),
                        help="Initialization model.")
    parser.add_argument("--random_init", type=bool, action="store_true", default=False,
                        help="Random initialize the model.")

    # task
    parser.add_argument("--task_dir", type=str, default="../../tasks",
                        help="The directory where tasks store.")
    parser.add_argument("--task_split_config", type=str, default="default",
                        help="The task split configuration, see `../../tasks/split/` for details.")

    # train, valid and test procedure
    parser.add_argument("--only_eval", action="store_true", default=False,
                        help="Whether to only perform testing procedure.")

    # hyper parameters
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--num_epochs", type=int, default=None,
                        help="Number of total training epochs.")
    parser.add_argument("--train_batch_size", type=int, default=None,
                        help="Size of training batch, per device.")
    parser.add_argument("--eval_batch_size", type=int, default=None,
                        help="Size of validation/testing batch, per device.")

    parser.add_argument("--max_source_length", type=int, default=None,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_source_pair_length", type=int, default=None,
                        help="The maximum total source pair sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", type=int, default=None,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm, 0 to disable.")
    parser.add_argument("--num_warmup_steps", type=int, default=None,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear",
                        help="The scheduler type to use.",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial",
                                 "constant", "constant_with_warmup"])

    parser.add_argument("--num_beams", type=int, default=10,
                        help="beam size for beam search.")
    parser.add_argument("--label_smoothing_factor", type=float, default=0.0,
                        help="Label smoothing factor.")

    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed, -1 to disable.")

    # environment
    parser.add_argument("--cuda_visible_devices", type=str, default=None,
                        help='Index (Indices) of the GPU to use in a cluster.')
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable cuda, overrides cuda_visible_devices.")
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"],
                        help="Mixed precision option, chosen from `no`, `fp16`, `bf16`")

    # ablation
    parser.add_argument("--training_sample", type=float, default=None,
                        help="Whether to sample a specific ratio (when between 0 and 1) or number (when >=0) "
                             "of training instance for training.")
    parser.add_argument("--train_from_scratch", action="store_true", default=False,
                        help="Whether to fine-tune from scratch, will not load pre-trained models.")

    # outputs and savings
    parser.add_argument("--run_name", type=str, default=None,
                        help="Unique name of current running, will be automatically set if it is None.")
    parser.add_argument("--wandb_mode", type=str, default="online",
                        choices=["online", "offline", "disabled"],
                        help="Set the wandb mode.")


def check_args(args):
    """Check if args values are valid, and conduct some default settings."""

    # task major metric
    args.major_metric = configs.TASK_TO_MAJOR_METRIC[args.task]
    # task type
    args.task_type = configs.TASK_NAME_TO_TYPE[args.task]

    # dataset
    dataset_list = configs.TASK_TO_DATASET[args.task]
    assert len(dataset_list) != 0, f'There is no dataset configured as the dataset of `{args.task}`.'
    if args.dataset is None:
        if len(dataset_list) > 1:
            raise ValueError(f"Please specific a dataset of task `{args.task}` "
                             f"when more than one datasets is configured.")
        else:
            args.dataset = dataset_list[0]
    else:
        assert args.dataset in dataset_list, \
            f'Dataset `{args.dataset}` is not configured as the dataset of task `{args.task}`.'

    # subset
    if args.subset is None:
        assert args.dataset not in configs.DATASET_TO_SUBSET, \
            f"Please specific a subset of dataset `{args.dataset}` when it has multiple subsets."
    else:
        assert args.dataset in configs.DATASET_TO_SUBSET, \
            f"Dataset `{args.dataset}` has no subset."
        assert args.subset in configs.DATASET_TO_SUBSET[args.dataset], \
            f"Dataset `{args.dataset}` has not subset called `{args.subset}`"

    # number of labels for classification tasks
    if args.task in configs.TASK_TYPE_TO_LIST["classification"]:
        if args.task == "exception":
            args.num_labels = 20
        else:
            args.num_labels = 2
    else:
        args.num_labels = 1

    # set language
    args.source_lang = None
    args.target_lang = None
    if args.dataset == "devign":
        args.source_lang = "c"
        args.target_lang = None
    elif args.dataset == "bigclonebench":
        args.source_lang = "java"
    elif args.dataset == "exception":
        args.source_lang = "python"
    elif args.dataset == "poj104":
        args.source_lang = "c"
        args.target_lang = "c"
    elif args.dataset == "advtest":
        args.source_lang = "en"
        args.target_lang = "python"
    elif args.dataset == "cosqa":
        args.source_lang = "python"
        args.target_lang = "en"
    elif args.dataset == "codetrans":
        args.source_lang, args.target_lang = args.subset.split("-")
    elif args.dataset == "bfp":
        args.source_lang = "java"
        args.target_lang = "java"
    elif args.dataset == "mutant":
        args.source_lang = "java"
        args.target_lang = "java"
    elif args.dataset == "assert":
        args.source_lang = "java"
        args.target_lang = "java"
    elif args.dataset == "codesearchnet":
        args.source_lang = args.subset
        args.target_lang = "en"
    elif args.dataset == "concode":
        args.source_lang = "en"
        args.target_lang = "c"
