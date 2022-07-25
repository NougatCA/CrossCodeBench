import torch
import argparse
import logging
import os
import time
import sys
from prettytable import PrettyTable
import random
import numpy as np
import wandb

import configs
from args import add_args, check_args
from utils import get_run_name, get_short_run_name
from run_tuning import run_tuning


def main():
    # parse arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.register("type", "bool", lambda v: v.lower() in ["yes", "true", "t", "1", "y"])

    add_args(parser)
    args = parser.parse_args()

    # check args
    # check_args(args)

    # prepare some preliminary arguments
    if args.run_name is None:
        args.run_name = get_run_name(args)
        args.short_run_name = get_short_run_name(args)
    args.run_name = "{}_{}".format(args.run_name, time.strftime("%Y%m%d_%H%M%S", time.localtime()))

    # outputs and savings
    args.output_dir = os.path.join("..", "..", "outputs", args.run_name)  # root of outputs/savings
    args.model_dir = os.path.join(args.output_dir, "models")  # dir of saving models
    args.eval_dir = os.path.join(args.output_dir, "evaluations")  # dir of saving evaluation results
    args.tb_dir = os.path.join(args.output_dir, "runs")  # dir of tracking running with tensorboard
    args.wandb_dir = os.path.join(args.output_dir, "wandb") # dir of tracking running with wandb
    for d in [args.model_dir, args.eval_dir, args.tb_dir, args.wandb_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    # logging, log to both console and file, log debug-level to file
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)
    # terminal printing
    console = logging.StreamHandler()
    console.setLevel(level=logging.INFO)
    logger.addHandler(console)
    # logging file
    file = logging.FileHandler(os.path.join(args.output_dir, "logging.log"))
    file.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s | %(filename)s | line %(lineno)d] - %(levelname)s: %(message)s")
    file.setFormatter(formatter)
    logger.addHandler(file)

    logger.info("=" * 20 + " INITIALIZING " + "=" * 20)

    # set distribution and mixed precision, using `accelerate` package
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    args.use_cuda = torch.cuda.is_available() and not args.no_cuda
    logger.info(f"Use cuda: {args.use_cuda}")
    if args.use_cuda:
        if args.cuda_visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
            logger.info(f"Devices: {args.cuda_visible_devices}")
        else:
            logger.info(f"Use all devices")

    # set random seed
    if args.random_seed > 0:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)

    # model identifier
    args.model_name = configs.model_to_ids[args.init_model]

    # log command and configs
    logger.debug("COMMAND: {}".format(" ".join(sys.argv)))

    config_table = PrettyTable()
    config_table.field_names = ["Configuration", "Value"]
    config_table.align["Configuration"] = "l"
    config_table.align["Value"] = "l"
    for config, value in vars(args).items():
        config_table.add_row([config, str(value)])
    logger.debug("Configurations:\n{}".format(config_table))

    # init wandb
    with open("wandb_api.key", mode="r", encoding="utf-8") as f:
        os.environ["WANDB_API_KEY"] = f.read().strip()
    run = wandb.init(project="Code-Instructions",
                     dir=args.wandb_dir,
                     name=args.short_run_name,
                     mode=args.wandb_mode,
                     config=vars(args))

    run_tuning(args, run)


if __name__ == "__main__":
    main()
