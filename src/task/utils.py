
from dataclasses import dataclass
import os
import json
from tqdm import tqdm


@dataclass
class DataInstance:
    inputs: list[str]
    outputs: str
    split: str
    idx: str


def read_devign_dataset(data_dir):

    instances = []
    for split in ["train", "valid", "test"]:
        with open(os.path.join(data_dir, f"{split}.jsonl"), mode="r", encoding="utf-8") as f:
            for line in f.readlines():
                js = json.loads(line.strip())
                label = "True" if js["target"] == 1 else "False"
                instances.append(
                    DataInstance(
                        inputs=js["func"],
                        outputs=label,
                        split=split,
                        idx=js["idx"]
                    )
                )
    return instances


def read_bigclonebench_dataset(data_dir):
    aux_data = {}
    with open(os.path.join(data_dir, "data.jsonl"), mode="r", encoding="utf-8") as f:
        for line in f.readlines():
            js = json.loads(line.strip())
            code = ' '.join(js['func'].split())
            aux_data[js['idx']] = code

    instances = []
    for split in ["train", "valid", "test"]:
        with open(os.path.join(data_dir, f"{split}.txt"), mode="r", encoding="utf-8") as f:
            lines = f.readlines()
            for idx, line in enumerate(tqdm(lines, desc="Reading", total=len(lines))):
                url1, url2, label = line.strip().split('\t')
                if url1 not in aux_data or url2 not in aux_data:
                    continue
                label = "True" if label.strip() == "1" else "False"
                instances.append(
                    DataInstance(
                        inputs=[aux_data[url1], aux_data[url2]],
                        outputs=label,
                        split=split,
                        idx=str(idx)
                    )
                )
    return instances
