import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import json
import os
import random
from dataclasses import dataclass
from typing import Union
import logging
from tqdm import tqdm
import multiprocessing
import multiprocess
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


class CodeDataset(Dataset):
    def __init__(self, data):
        super(CodeDataset, self).__init__()
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def load_instances(args, split):
    assert split in ["tune", "eval"]

    # load split config
    split_config_path = os.path.join(args.task_dir, "split", f"{args.task_split_config}.json")
    logger.info(f"Start loading task split configuration from '{split_config_path}'")
    with open(split_config_path, mode="r", encoding="utf-8") as f:
        split_config = json.load(f)
        task_names = split_config[split]

    if split == "tune" and 0 < args.max_num_tune_tasks < len(task_names):
        task_names = task_names[:args.max_num_tune_tasks]

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
                with open(os.path.join(args.task_dir, f"{task_name}.meta.json"),
                          mode="r", encoding="utf-8") as meta_f, \
                        open(os.path.join(args.task_dir, f"{task_name}.data.json"),
                             mode="r", encoding="utf-8") as data_f:
                    meta = json.load(meta_f)
                    data = json.load(data_f)
            except MemoryError:
                logger.error(f"Run out of memory when loading data file of task '{task_name}', skipped.")
                continue
            # get the instances
            task_instances = data["Instances"]
            # sample
            if split == "tune" and len(task_instances) > args.max_sample_per_task:
                # task_instances = random.sample(task_instances, k=args.max_sample_per_task)
                task_instances = task_instances[:args.max_sample_per_task]
            elif split == "eval" and len(task_instances) > args.max_eval_sample_per_task:
                # task_instances = random.sample(task_instances, k=args.max_eval_sample_per_task)
                task_instances = task_instances[:args.max_eval_sample_per_task]
            # task id
            task_id = task_name.split("_")[1]
            meta["task_id"] = task_id

            for instance in task_instances:
                instances.append(
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
                                use_few_shot,
                                num_shots,
                                use_prompt,
                                use_instruction,
                                instruction_items,
                                num_pos_examples,
                                num_neg_examples,
                                max_source_length,
                                max_target_length,
                                max_instruction_length) -> InputFeature:
    def convert_inputs_to_seq(inputs: Union[str, List[str]]):
        return inputs if isinstance(inputs, str) else " ||| ".join(instance.inputs)

    # build input ids
    # use few-shot
    if use_few_shot:
        description = "{}: ".format(instance.meta["Prompt"][0])

        examples = random.sample(instance.meta["Positive Examples"], k=num_shots)
        few_shot_txt = "; ".join(["Example {} - input: {}; output: {}".format(
            idx + 1,
            convert_inputs_to_seq(example["input"]),
            convert_inputs_to_seq(example["output"])) for idx, example in enumerate(examples)])
        few_shot_txt += f"; Now complete the following example − input: "
        few_shot_txt = description + few_shot_txt

        source_txt = f"{convert_inputs_to_seq(instance.inputs)}; output:"

        few_shot_ids = tokenizer.encode(few_shot_txt,
                                        padding="max_length",
                                        max_length=max_instruction_length,
                                        truncation=True,
                                        add_special_tokens=False)
        source_ids = tokenizer.encode(source_txt,
                                      padding="max_length",
                                      max_length=max_source_length,
                                      truncation=True)

        input_ids = few_shot_ids + source_ids
    # use prompt
    elif use_prompt:
        input_txt = "{}: {}".format(instance.meta["Prompt"][0], convert_inputs_to_seq(instance.inputs))
        input_ids = tokenizer.encode(input_txt,
                                     padding="max_length",
                                     max_length=max_instruction_length + max_source_length,
                                     truncation=True)
    # use task instruction
    elif use_instruction:
        instruction_keys = instruction_items.split("|")
        instruction_seqs = []
        for key in instruction_keys:
            values = instance.meta[key]
            item = "{}: {}".format(key, ", ".join(values))
            instruction_seqs.append(item)
        instruction_txt = "; ".join(instruction_seqs)
        if num_pos_examples > 0:
            pos_examples = random.sample(instance.meta["Positive Examples"], k=num_pos_examples)
            for pos_id, pos_example in enumerate(pos_examples):
                instruction_txt += "; Positive Example {} - input: {}; output: {}, explanation: {}".format(
                    pos_id + 1,
                    convert_inputs_to_seq(pos_example["input"]),
                    convert_inputs_to_seq(pos_example["output"]),
                    pos_example["reason"])
        if num_neg_examples > 0:
            neg_examples = random.sample(instance.meta["Negative Examples"], k=num_neg_examples)
            for neg_id, neg_example in enumerate(neg_examples):
                instruction_txt += "; Negative Example {} - input: {}; output: {}, explanation: {}".format(
                    neg_id + 1,
                    convert_inputs_to_seq(neg_example["input"]),
                    convert_inputs_to_seq(neg_example["output"]),
                    neg_example["reason"])

        instruction_txt += f"; Now complete the following example − input: "

        instruction_ids = tokenizer.encode(instruction_txt,
                                           padding="max_length",
                                           max_length=max_instruction_length,
                                           truncation=True,
                                           add_special_tokens=False)

        source_txt = f"{convert_inputs_to_seq(instance.inputs)}; output:"
        source_ids = tokenizer.encode(source_txt,
                                      padding="max_length",
                                      max_length=max_source_length,
                                      truncation=True)

        input_ids = instruction_ids + source_ids
    # no verbalizer
    else:
        input_ids = tokenizer.encode(convert_inputs_to_seq(instance.inputs),
                                     padding="max_length",
                                     max_length=max_instruction_length + max_source_length,
                                     truncation=True)

    decoder_input_ids = tokenizer.encode(convert_inputs_to_seq(instance.outputs),
                                         padding="max_length",
                                         max_length=max_target_length,
                                         truncation=True)
    return InputFeature(input_ids=input_ids, decoder_input_ids=decoder_input_ids)


def create_dataset(args, instances, tokenizer):
    """Create dataset by converting examples to input features."""

    logger.info(f"Start encoding instances into features")
    processes = multiprocessing.cpu_count() // 2 if not args.single_thread else 1
    logger.info(f"Using {processes} processes to encode instances")
    encode_func = partial(convert_instance_to_feature,
                          tokenizer=tokenizer,
                          use_few_shot=args.use_few_shot,
                          num_shots=args.num_shots,
                          use_prompt=args.use_prompt,
                          use_instruction=args.use_instruction,
                          instruction_items=args.instruction_items,
                          num_pos_examples=args.num_pos_examples,
                          num_neg_examples=args.num_neg_examples,
                          max_source_length=args.max_source_length,
                          max_target_length=args.max_target_length,
                          max_instruction_length=args.max_instruction_length)
    if processes > 1:
        # with multiprocessing.get_context("spawn").Pool(processes=processes) as p:
        #     features = list(p.map(encode_func, tqdm(instances, total=len(instances), desc="Encoding")))
        pool = multiprocess.Pool(processes=processes)
        # pool = multiprocess.get_context("spawn").Pool(processes=processes)
        features = pool.map(encode_func, tqdm(instances, total=len(instances), desc="Encoding"))
        pool.close()
        pool.join()
    else:
        features = [encode_func(example) for example in tqdm(instances, total=len(instances), desc="Encoding")]

    logger.info(f"Features are prepared, start building the dataset.")
    input_dicts = []
    for f in tqdm(features, total=len(features), desc="Building"):
        input_ids = torch.tensor(f.input_ids, dtype=torch.long)
        decoder_input_ids = torch.tensor(f.decoder_input_ids, dtype=torch.long)
        input_dicts.append(
            {
                "input_ids": input_ids,
                "attention_mask": input_ids.ne(tokenizer.pad_token_id),
                "labels": decoder_input_ids,
                "decoder_attention_mask": decoder_input_ids.ne(tokenizer.pad_token_id)
            }
        )
    dataset = CodeDataset(input_dicts)
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

    return dataloader
