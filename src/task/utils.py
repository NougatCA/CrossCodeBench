
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


def read_devign(data_dir):

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


def read_bigclonebench(data_dir):

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


def read_exception_type(data_dir):

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


def read_function_docstring_mismatch(data_dir):

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
            doc = js["docstring"]
            target_txt = js["label"]
            instances.append(
                DataInstance(
                    inputs=[code, doc],
                    outputs=target_txt,
                    split=split,
                    idx=str(idx)
                )
            )
            sizes[split] += 1
    sizes["total"] = len(instances)
    return instances, sizes


def read_variable_misuse(data_dir):

    data_dir = os.path.join(data_dir, "variable_misuse")

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


def read_swapped_operands(data_dir):

    data_dir = os.path.join(data_dir, "swapped_operands")

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


def read_wrong_binary_operator(data_dir):

    data_dir = os.path.join(data_dir, "wrong_binary_operator")

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


def read_web_query_test(data_dir):

    data_dir = os.path.join(data_dir, "web_query_test")

    instances = []
    sizes = {
        "test": 0
    }

    idx_to_answer = {}
    with open(os.path.join(data_dir, "answers.txt"), mode="r", encoding="utf-8") as f:
        for line in f:
            idx, answer = line.strip().split("\t")
            idx_to_answer[idx] = int(answer)

    with open(os.path.join(data_dir, "test_webquery.json")) as f:
        data = json.load(f)
    for js in tqdm(data, total=len(data), desc=f"Reading"):
        code = js["code"]
        nl = js["doc"]
        idx = js["idx"]
        label = "Yes" if idx_to_answer[idx] == 1 else "No"
        instances.append(
            DataInstance(
                inputs=[nl, code],
                outputs=label,
                split="test",
                idx=idx
            )
        )
        sizes["test"] += 1
    sizes["total"] = len(instances)
    return instances, sizes


def read_cosqa(data_dir):

    data_dir = os.path.join(data_dir, "cosqa")

    instances = []
    sizes = {
        "train": 0,
        "valid": 0,
    }

    for split in ["train", "valid"]:
        with open(os.path.join(data_dir, f"cosqa-{split}.json"), mode="r", encoding="utf-8") as f:
            data = json.load(f)
        for js in tqdm(data, total=len(data), desc=f"Reading"):
            code = js["code"]
            nl = js["doc"]
            idx = js["idx"]
            label = "Yes" if js["label"] == 1 else "No"
            instances.append(
                DataInstance(
                    inputs=[nl, code],
                    outputs=label,
                    split=split,
                    idx=idx
                )
            )
            sizes[split] += 1
    assert sum(sizes.values()) == len(instances)
    sizes["total"] = len(instances)
    return instances, sizes


def read_code_trans_java_cs(data_dir):

    data_dir = os.path.join(data_dir, "code_trans")

    instances = []
    sizes = {
        "train": 0,
        "valid": 0,
        "test": 0
    }
    for split in ["train", "valid", "test"]:
        with open(os.path.join(data_dir, f"{split}.java-cs.txt.java"), mode="r", encoding="utf-8") as src_f, \
             open(os.path.join(data_dir, f"{split}.java-cs.txt.cs"), mode="r", encoding="utf-8") as tgt_f:
            sources = src_f.readlines()
            targets = tgt_f.readlines()
        assert len(sources) == len(targets)
        for idx, (source, target) in enumerate(tqdm(zip(sources, targets), desc="Reading", total=len(sources))):
            instances.append(
                DataInstance(
                    inputs=source,
                    outputs=target,
                    split=split,
                    idx=str(idx)
                )
            )
            sizes[split] += 1
    assert sum(sizes.values()) == len(instances)
    assert sum(sizes.values()) == len(instances)
    sizes["total"] = len(instances)
    return instances, sizes


def read_code_trans_cs_java(data_dir):

    data_dir = os.path.join(data_dir, "code_trans")

    instances = []
    sizes = {
        "train": 0,
        "valid": 0,
        "test": 0
    }
    for split in ["train", "valid", "test"]:
        with open(os.path.join(data_dir, f"{split}.java-cs.txt.cs"), mode="r", encoding="utf-8") as src_f, \
             open(os.path.join(data_dir, f"{split}.java-cs.txt.java"), mode="r", encoding="utf-8") as tgt_f:
            sources = src_f.readlines()
            targets = tgt_f.readlines()
        assert len(sources) == len(targets)
        for idx, (source, target) in enumerate(tqdm(zip(sources, targets), desc="Reading", total=len(sources))):
            instances.append(
                DataInstance(
                    inputs=source,
                    outputs=target,
                    split=split,
                    idx=str(idx)
                )
            )
            sizes[split] += 1
    assert sum(sizes.values()) == len(instances)
    assert sum(sizes.values()) == len(instances)
    sizes["total"] = len(instances)
    return instances, sizes
