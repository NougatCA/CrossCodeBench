import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader
import json
import os
import random
from dataclasses import dataclass
from typing import Union
import logging
from tqdm import tqdm
import multiprocessing
from functools import partial
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class DataInstance:
    inputs: Union[str, list[str]]
    outputs: Union[str, list[str]]
    split: str
    idx: str
    meta: dict


@dataclass
class InputFeature:
    input_ids: List[int]
    decoder_input_ids: List[int]


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
            # try to load meta and data from json file
            try:
                with open(os.path.join(args.task_dir, f"{task_name}.meta.json"), mode="r", encoding="utf-8") as meta_f, \
                        open(os.path.join(args.task_dir, f"{task_name}.data.json"), mode="r", encoding="utf-8") as data_f:
                    meta = json.load(meta_f)
                    data = json.load(data_f)
            except MemoryError:
                logger.error(f"Run out of memory when loading data file of task '{task_name}', skipped.")
                continue
            # get the instances
            task_instances = data["Instances"]
            # sample
            if len(task_instances) > args.max_sample_per_task:
                task_instances = random.sample(task_instances, k=args.max_sample_per_task)

            for instance in task_instances:
                instance.append(
                    DataInstance(
                        inputs=instance["input"],
                        outputs=instance["output"],
                        split=instance["split"],
                        idx=instance["idx"],
                        meta=meta
                    )
                )

            logger.debug(f"Finish loading '{task_name}', size: {len(task_instances)}.")
        else:
            logger.warning(f"WARNING: Task '{task_name}' is skipped as it not presents "
                           f"in the given task directory: '{args.task_dir}'")

    logger.info(f"Data loaded, total size: {len(instances)}")

    return instances


def convert_instance_to_feature(instance: DataInstance,
                                tokenizer,
                                max_source_length,
                                max_target_length,
                                max_instruction_length) -> InputFeature:
    pass



def create_dataset(args, instances, tokenizer):
    """Create dataset by converting examples to input features."""

    logger.info(f"Start encoding instances into features")
    processes = multiprocessing.cpu_count()
    encode_func = partial(convert_instance_to_feature,
                          tokenizer=tokenizer,
                          max_source_length=args.max_source_length,
                          max_target_length=args.max_target_length)
    if processes > 1:
        with multiprocessing.Pool(processes=processes) as p:
            features = list(p.map(encode_func, tqdm(instances, total=len(instances), desc="Encoding")))
    else:
        features = [encode_func(example) for example in tqdm(instances, total=len(instances), desc="Encoding")]

    all_input_ids, all_decoder_input_ids = [], []
    for f in features:
        all_input_ids.append(f.input_ids)
        all_decoder_input_ids.append(f.decoder_input_ids)
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_decoder_input_ids = torch.tensor(all_decoder_input_ids, dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_decoder_input_ids)
    return dataset


def prepare_data(args, split, tokenizer):
    assert split in ["tune", "eval"]
    instances = load_instances(args, split=split)
    dataset = create_dataset(args, instances, tokenizer)
    dataloader = DataLoader(dataset,
                            shuffle=True if split == "tune" else False,
                            batch_size=args.train_batch_size if split == "tune" else args.eval_batch_size,
                            num_workers=4,
                            pin_memory=True)

    return dataset, dataloader
