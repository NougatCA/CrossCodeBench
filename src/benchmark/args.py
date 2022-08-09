from transformers import SchedulerType
from argparse import ArgumentParser
import logging

import configs


logger = logging.getLogger(__name__)


def add_args(parser: ArgumentParser):

    # model
    parser.add_argument("--init_model", type=str, default="codet5",
                        choices=configs.model_to_ids.keys(),
                        help="Initialization model.")
    parser.add_argument("--random_init", action="store_true", default=False,
                        help="Random initialize the model.")

    # task
    parser.add_argument("--task_dir", type=str, default="../../tasks",
                        help="The directory where tasks store.")
    parser.add_argument("--task_split_config", type=str, default="translation",
                        help="The task split configuration, see `../../tasks/split/` for details.")
    parser.add_argument("--max_num_tune_tasks", type=int, default=0,
                        help="Maximum number of tuning tasks, 0 to use all.")

    # train, valid and test procedure
    parser.add_argument("--only_eval", action="store_true", default=False,
                        help="Whether to only perform testing procedure.")

    # hyper parameters
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--num_epochs", type=int, default=2,
                        help="Number of total training epochs.")
    parser.add_argument("--train_batch_size", type=int, default=32,
                        help="Size of training batch, per device.")
    parser.add_argument("--eval_batch_size", type=int, default=8,
                        help="Size of validation/testing batch, per device.")

    parser.add_argument("--max_source_length", type=int, default=256,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", type=int, default=256,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_instruction_length", type=int, default=768,
                        help="The maximum total instruction sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight deay if we apply some.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm, 0 to disable.")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear",
                        help="The scheduler type to use.",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial",
                                 "constant", "constant_with_warmup"])

    parser.add_argument("--num_beams", type=int, default=5,
                        help="beam size for beam search.")
    parser.add_argument("--label_smoothing_factor", type=float, default=0.0,
                        help="Label smoothing factor.")

    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed, -1 to disable.")

    # environment
    parser.add_argument("--cuda_visible_devices", type=str, default=None,
                        help='Index (Indices) of the GPU to use in a cluster.')
    parser.add_argument("--no_cuda", action="store_true", default=False,
                        help="Disable cuda, overrides cuda_visible_devices.")
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"],
                        help="Mixed precision option, chosen from `no`, `fp16`, `bf16`.")
    parser.add_argument("--single_thread", action="store_true", default=False,
                        help="Whether to use single thread to encode examples.")

    # ablation
    parser.add_argument("--max_sample_per_task", type=int, default=10000,
                        help="Sample a number of training instance per task for training.")
    parser.add_argument("--max_eval_sample_per_task", type=int, default=500,
                        help="Maximum number of samples per task for evaluating.")

    # verbalizer type, default by none
    # few-shot and one-shot
    parser.add_argument("--use_few_shot", action="store_true", default=False,
                        help="Whether to use few-shot learning.")
    parser.add_argument("--num_shots", type=int, default=1,
                        choices=[1, 2, 3, 4],
                        help="Number of examples in the few-shot learning.")
    # prompt
    parser.add_argument("--use_prompt", action="store_true", default=False,
                        help="Whether to use prompt.")
    # task instruction
    parser.add_argument("--use_instruction", action="store_true", default=False,
                        help="Whether to use task instruction.")
    parser.add_argument("--instruction_items", type=str, default=None,
                        help="Items used in the task instruction, separated by '|'.")
    parser.add_argument("--num_pos_examples", type=int, default=2,
                        choices=[0, 1, 2, 3, 4],
                        help="Number of positive examples in the instructions.")
    parser.add_argument("--num_neg_examples", type=int, default=2,
                        choices=[0, 1, 2, 3, 4],
                        help="Number of negative examples in the instructions.")

    # outputs and savings
    parser.add_argument("--run_name", type=str, default=None,
                        help="Unique name of current running, will be automatically set if it is None.")
    parser.add_argument("--wandb_mode", type=str, default="online",
                        choices=["online", "offline", "disabled"],
                        help="Set the wandb mode.")


def check_args(args):
    """Check if args values are valid, and conduct some default settings."""
    if args.use_few_shot:
        if args.num_shots > 1:
            logger.info("Verbalizer: few-shot")
            logger.info(f"# of examples: {args.num_shots}")
        else:
            logger.info("Verbalizer: one-shot")
    elif args.use_prompt:
        logger.info("Verbalizer: prompt")
    elif args.use_instruction:
        logger.info("Verbalizer: task instruction")
        if args.instruction_items:
            valid_items = []
            items = args.instruction_items.split("|")
            for item in items:
                if item not in configs.all_instruction_items:
                    logger.warning(f"The item '{item}' is not in the instruction keys, skipped.")
                else:
                    valid_items.append(item)
            args.instruction_items = "|".join(valid_items)
            logger.info("Instruction items: {}".format(", ".join(valid_items)))
        else:
            args.instruction_items = "|".join(configs.all_instruction_items)
            logger.info("Use all instruction items")
        logger.info(f"# of positive/negative examples: {args.num_pos_examples}/{args.num_neg_examples}")
    else:
        logger.info("No verbalizer")
