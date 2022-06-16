
from dataclasses import dataclass
import os
import json
from tqdm import tqdm
from typing import Union


@dataclass
class DataInstance:
    inputs: Union[str, list[str]]
    outputs: str
    split: str
    idx: str


def read_devign_dataset(data_dir):

    data_dir = os.path.join(data_dir, "devign")

    instances = []
    sizes = {
        "train": 0,
        "valid": 0,
        "test": 0
    }
    for split in ["train", "valid", "test"]:
        with open(os.path.join(data_dir, f"{split}.jsonl"), mode="r", encoding="utf-8") as f:
            for line in f.readlines():
                js = json.loads(line.strip())
                label = "Yes" if js["target"] == 1 else "No"
                instances.append(
                    DataInstance(
                        inputs=js["func"],
                        outputs=label,
                        split=split,
                        idx=js["idx"]
                    )
                )
                sizes[split] += 1
    sizes["total"] = len(instances)
    return instances, sizes


def read_bigclonebench_dataset(data_dir):

    data_dir = os.path.join(data_dir, "bigclonebench")

    aux_data = {}
    with open(os.path.join(data_dir, "data.jsonl"), mode="r", encoding="utf-8") as f:
        for line in f.readlines():
            js = json.loads(line.strip())
            code = ' '.join(js['func'].split())
            aux_data[js['idx']] = code

    instances = []
    sizes = {
        "train": 0,
        "valid": 0,
        "test": 0
    }
    for split in ["train", "valid", "test"]:
        with open(os.path.join(data_dir, f"{split}.txt"), mode="r", encoding="utf-8") as f:
            lines = f.readlines()
        for idx, line in enumerate(tqdm(lines, desc="Reading", total=len(lines))):
            url1, url2, label = line.strip().split('\t')
            if url1 not in aux_data or url2 not in aux_data:
                continue
            label = "Yes" if label.strip() == "1" else "No"
            instances.append(
                DataInstance(
                    inputs=[aux_data[url1], aux_data[url2]],
                    outputs=label,
                    split=split,
                    idx=str(idx)
                )
            )
            sizes[split] += 1
    sizes["total"] = len(instances)
    return instances, sizes


def read_exception_type_dataset(data_dir):

    data_dir = os.path.join(data_dir, "exception_type")

    instances = []
    sizes = {
        "train": 0,
        "valid": 0,
        "test": 0
    }
    for split in ["train", "valid", "test"]:
        with open(os.path.join(data_dir, f"{split}.jsonl"), mode="r", encoding="utf-8") as f:
            lines = f.readlines()
        for idx, line in enumerate(tqdm(lines, desc="Reading", total=len(lines))):
            js = json.loads(line.strip())
            code = js["function"]
            target_txt = js["label"]
            instances.append(
                DataInstance(
                    inputs=code,
                    outputs=target_txt,
                    split=split,
                    idx=str(idx)
                )
            )
            sizes[split] += 1
    sizes["total"] = len(instances)
    return instances, sizes


def read_function_docstring_mismatch_dataset(data_dir):

    data_dir = os.path.join(data_dir, "function_docstring_mismatch")

    instances = []
    sizes = {
        "train": 0,
        "valid": 0,
        "test": 0
    }
    for split in ["train", "valid", "test"]:
        with open(os.path.join(data_dir, f"{split}.jsonl"), mode="r", encoding="utf-8") as f:
            lines = f.readlines()
        for idx, line in enumerate(tqdm(lines, desc="Reading", total=len(lines))):
            js = json.loads(line.strip())
            code = js["function"]
            target_txt = js["label"]
            instances.append(
                DataInstance(
                    inputs=code,
                    outputs=target_txt,
                    split=split,
                    idx=str(idx)
                )
            )
            sizes[split] += 1
    sizes["total"] = len(instances)
    return instances, sizes
