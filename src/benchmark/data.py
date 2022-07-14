import json
import os
from dataclasses import dataclass
from typing import Union
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class DataInstance:
    inputs: Union[str, list[str]]
    outputs: Union[str, list[str]]
    split: str
    idx: str


def load_instances(args, split):
    assert split in ["tune", "eval"]

    # load split config
    split_config_path = os.path.join(args.task_dir, "split", f"{args.task_split_config}.json")
    logger.info(f"Start loading task split configuration from '{split_config_path}'")
    with open(split_config_path, mode="r", encoding="utf-8") as f:
        split_config = json.load(f)
        task_names = split_config[split]

    # load data
    logger.info(f"Start loading '{split}' data from '{args.task_dir}'")

    instances = []
    for task_name in tqdm(task_names):
        meta_path = os.path.join(args.task_dir, f"{task_name}.meta.json")
        data_path = os.path.join(args.task_dir, f"{task_name}.data.json")
        if os.path.exists(meta_path) and os.path.exists(data_path):
            logger.debug(f"Start loading '{task_name}'.")
            with open(os.path.join(args.task_dir, f"{task_name}.meta.json"), mode="r", encoding="utf-8") as meta_f, \
                    open(os.path.join(args.task_dir, f"{task_name}.data.json"), mode="r", encoding="utf-8") as data_f:
                meta_data = json.load(meta_f)
                data = json.load(data_f)
            logger.debug(f"Finish loading '{task_name}'.")
        else:
            logger.warning(f"WARNING: Task '{task_name}' is skipped as it not presents "
                           f"in the given task directory: '{args.task_dir}'")

    logger.info(f"{split} data loaded, total size: {len(examples)}")

    # sample specific number/ratio of examples if needed
    if args.training_sample is not None and args.training_sample > 0:
        if args.training_sample < 1:
            num_to_sample = int(len(examples) * args.training_sample)
            examples = random.sample(examples, num_to_sample)
        elif args.training_sample >= 1:
            examples = random.sample(examples, args.training_sample)
        logger.info(f"Sampled {len(examples)} data because '--training-sample={args.training_sample}'")

    return examples


def prepare_data(args, split):
    assert split in ["tune", "eval"]
    instances = load_instances(args, split=split)