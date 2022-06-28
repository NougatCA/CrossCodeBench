
from dataclasses import dataclass
import os
import json
from tqdm import tqdm
from typing import Union


@dataclass
class DataInstance:
    inputs: Union[str, list[str]]
    outputs: Union[str, list[str]]
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
                    inputs=source.strip(),
                    outputs=target.strip(),
                    split=split,
                    idx=str(idx)
                )
            )
            sizes[split] += 1
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
                    inputs=source.strip(),
                    outputs=target.strip(),
                    split=split,
                    idx=str(idx)
                )
            )
            sizes[split] += 1
    assert sum(sizes.values()) == len(instances)
    sizes["total"] = len(instances)
    return instances, sizes


def read_trans_coder(data_dir, source_lang, target_lang):
    assert source_lang in ["java", "python", "cpp"]
    assert target_lang in ["java", "python", "cpp"]

    data_dir = os.path.join(data_dir, "trans_coder")
    instances = []
    sizes = {
        "valid": 0,
        "test": 0
    }
    for split in ["valid", "test"]:
        with open(os.path.join(data_dir, f"transcoder_{split}.{source_lang}.tok"), mode="r", encoding="utf-8") as src_f, \
             open(os.path.join(data_dir, f"transcoder_{split}.{target_lang}.tok"), mode="r", encoding="utf-8") as tgt_f:
            sources = src_f.readlines()
            targets = tgt_f.readlines()
        assert len(sources) == len(targets)
        for idx, (source, target) in enumerate(tqdm(zip(sources, targets), desc="Reading", total=len(sources))):
            src_idx, source = source.split(" | ", 1)
            src_idx = src_idx.strip()
            source = source.strip()

            tgt_idx, target = target.split(" | ", 1)
            tgt_idx = tgt_idx.strip()
            target = target.strip()

            assert src_idx == tgt_idx

            instances.append(
                DataInstance(
                    inputs=source,
                    outputs=target,
                    split=split,
                    idx=src_idx
                )
            )
            sizes[split] += 1
    assert sum(sizes.values()) == len(instances)
    sizes["total"] = len(instances)
    return instances, sizes


def read_bfp(data_dir, subset):
    assert subset in ["small", "medium"]
    data_dir = os.path.join(data_dir, "bfp", subset)

    instances = []
    sizes = {
        "train": 0,
        "valid": 0,
        "test": 0
    }
    for split in ["train", "valid", "test"]:
        with open(os.path.join(data_dir, f"{split}.buggy-fixed.buggy"), mode="r", encoding="utf-8") as src_f, \
             open(os.path.join(data_dir, f"{split}.buggy-fixed.fixed"), mode="r", encoding="utf-8") as tgt_f:
            sources = src_f.readlines()
            targets = tgt_f.readlines()
        assert len(sources) == len(targets)
        for idx, (source, target) in enumerate(tqdm(zip(sources, targets), desc="Reading", total=len(sources))):
            instances.append(
                DataInstance(
                    inputs=source.strip(),
                    outputs=target.strip(),
                    split=split,
                    idx=str(idx)
                )
            )
            sizes[split] += 1
    assert sum(sizes.values()) == len(instances)
    sizes["total"] = len(instances)
    return instances, sizes


def read_any_code_completion(data_dir):
    data_dir = os.path.join(data_dir, "any_code_completion")

    instances = []
    sizes = {
        "train": 0,
        "valid": 0,
        "test": 0
    }
    for split in ["train", "valid", "test"]:
        with open(os.path.join(data_dir, f"data.TargetType.seq.{split}.source.txt"), mode="r", encoding="utf-8") as src_f, \
             open(os.path.join(data_dir, f"data.TargetType.seq.{split}.target.txt"), mode="r", encoding="utf-8") as tgt_f:
            sources = src_f.readlines()
            targets = tgt_f.readlines()
        assert len(sources) == len(targets)
        for idx, (source, target) in enumerate(tqdm(zip(sources, targets), desc="Reading", total=len(sources))):
            instances.append(
                DataInstance(
                    inputs=source.strip(),
                    outputs=target.strip(),
                    split=split,
                    idx=str(idx)
                )
            )
            sizes[split] += 1
    assert sum(sizes.values()) == len(instances)
    sizes["total"] = len(instances)
    return instances, sizes


def read_tp_mutant(data_dir):
    data_dir = os.path.join(data_dir, "tp_mutant")

    instances = []
    sizes = {
        "train": 0,
        "valid": 0,
        "test": 0
    }
    for split in ["train", "valid", "test"]:
        with open(os.path.join(data_dir, f"{split}_fixed.txt"), mode="r", encoding="utf-8") as src_f, \
                open(os.path.join(data_dir, f"{split}_buggy.txt"), mode="r", encoding="utf-8") as tgt_f:
            sources = src_f.readlines()
            targets = tgt_f.readlines()
        assert len(sources) == len(targets)
        for idx, (source, target) in enumerate(tqdm(zip(sources, targets), desc="Reading", total=len(sources))):
            instances.append(
                DataInstance(
                    inputs=source.strip(),
                    outputs=target.strip(),
                    split=split,
                    idx=str(idx)
                )
            )
            sizes[split] += 1
    assert sum(sizes.values()) == len(instances)
    sizes["total"] = len(instances)
    return instances, sizes


def read_tp_fixing(data_dir):
    data_dir = os.path.join(data_dir, "tp_mutant")

    instances = []
    sizes = {
        "train": 0,
        "valid": 0,
        "test": 0
    }
    for split in ["train", "valid", "test"]:
        with open(os.path.join(data_dir, f"{split}_buggy.txt"), mode="r", encoding="utf-8") as src_f, \
                open(os.path.join(data_dir, f"{split}_fixed.txt"), mode="r", encoding="utf-8") as tgt_f:
            sources = src_f.readlines()
            targets = tgt_f.readlines()
        assert len(sources) == len(targets)
        for idx, (source, target) in enumerate(tqdm(zip(sources, targets), desc="Reading", total=len(sources))):
            instances.append(
                DataInstance(
                    inputs=source.strip(),
                    outputs=target.strip(),
                    split=split,
                    idx=str(idx)
                )
            )
            sizes[split] += 1
    assert sum(sizes.values()) == len(instances)
    sizes["total"] = len(instances)
    return instances, sizes


def read_tap(data_dir, subset):
    assert subset in ["abs", "raw"]

    data_dir = os.path.join(data_dir, "tap", subset)

    instances = []
    sizes = {
        "train": 0,
        "valid": 0,
        "test": 0
    }
    for split in ["train", "valid", "test"]:
        with open(os.path.join(data_dir, f"{split}_methods.txt"), mode="r", encoding="utf-8") as src_f, \
                open(os.path.join(data_dir, f"{split}_assert.txt"), mode="r", encoding="utf-8") as tgt_f:
            sources = src_f.readlines()
            targets = tgt_f.readlines()
        assert len(sources) == len(targets)
        for idx, (source, target) in enumerate(tqdm(zip(sources, targets), desc="Reading", total=len(sources))):
            instances.append(
                DataInstance(
                    inputs=source.strip(),
                    outputs=target.strip(),
                    split=split,
                    idx=str(idx)
                )
            )
            sizes[split] += 1
    assert sum(sizes.values()) == len(instances)
    sizes["total"] = len(instances)
    return instances, sizes


def read_code_search_net_filtered(data_dir, subset):
    assert subset in ["java", "python", "javascript", "php", "go", "ruby"]

    data_dir = os.path.join(data_dir, "code_search_net_filtered", subset)

    instances = []
    sizes = {
        "train": 0,
        "valid": 0,
        "test": 0
    }
    for split in ["train", "valid", "test"]:
        with open(os.path.join(data_dir, f"{split}.jsonl"), mode="r", encoding="utf-8") as f:
            lines = f.readlines()
        for idx, line in enumerate(tqdm(lines, total=len(lines), desc=f"Loading {split} data")):
            js = json.loads(line.strip())
            source = " ".join(js["code_tokens"])
            target = " ".join(js["docstring_tokens"])
            instances.append(
                DataInstance(
                    inputs=source,
                    outputs=target,
                    split=split,
                    idx=js["path"]
                )
            )
            sizes[split] += 1
    assert sum(sizes.values()) == len(instances)
    sizes["total"] = len(instances)
    return instances, sizes


def read_funcom_tokenized(data_dir):
    data_dir = os.path.join(data_dir, "funcom", "tokenized")

    instances = []
    sizes = {
        "train": 0,
        "valid": 0,
        "test": 0
    }
    for split in ["train", "valid", "test"]:
        with open(os.path.join(data_dir, f"functions.{split}"), mode="r", encoding="utf-8") as src_f, \
                open(os.path.join(data_dir, f"comments.{split}"), mode="r", encoding="utf-8") as tgt_f:
            sources = src_f.readlines()
            targets = tgt_f.readlines()
        assert len(sources) == len(targets)
        for source, target in tqdm(zip(sources, targets), desc="Reading", total=len(sources)):
            src_idx, source = source.split("\t", 1)
            tgt_idx, target = target.split("\t", 1)
            assert src_idx == tgt_idx
            instances.append(
                DataInstance(
                    inputs=source.strip(),
                    outputs=target.strip(),
                    split=split,
                    idx=src_idx
                )
            )
            sizes[split] += 1
    assert sum(sizes.values()) == len(instances)
    sizes["total"] = len(instances)
    return instances, sizes


def read_funcom_raw(data_dir):
    data_dir = os.path.join(data_dir, "funcom", "raw")
    instances = []
    sizes = {
        "total": 0
    }
    with open(os.path.join(data_dir, "functions.json"), mode="r", encoding="utf-8") as f:
        fun_data = json.load(f)
    with open(os.path.join(data_dir, "comments.json"), mode="r", encoding="utf-8") as f:
        com_data = json.load(f)

    for k, v in fun_data.items():
        assert k in com_data
        function = v.strip()
        comment = com_data[k].strip()
        instances.append(
            DataInstance(
                inputs=function,
                outputs=comment,
                split="",
                idx=str(k)
            )
        )
        sizes["total"] += 1

    return instances, sizes


def read_deep_com(data_dir):
    data_dir = os.path.join(data_dir, "deep_com")

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
            code = js["code"].strip()
            doc = js["nl"].strip()
            instances.append(
                DataInstance(
                    inputs=code,
                    outputs=doc,
                    split=split,
                    idx=str(idx)
                )
            )
            sizes[split] += 1
    assert sum(sizes.values()) == len(instances)
    sizes["total"] = len(instances)
    return instances, sizes


def read_tl_code_sum(data_dir):
    data_dir = os.path.join(data_dir, "tl_code_sum")

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
            code = js["code"].strip()
            doc = js["comment"].strip()
            instances.append(
                DataInstance(
                    inputs=code,
                    outputs=doc,
                    split=split,
                    idx=str(js["id"])
                )
            )
            sizes[split] += 1
    assert sum(sizes.values()) == len(instances)
    sizes["total"] = len(instances)
    return instances, sizes


def read_tl_api_seq_sum(data_dir):
    data_dir = os.path.join(data_dir, "tl_code_sum")

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
            api_seq = js["api_seq"]
            doc = js["comment"].strip()
            instances.append(
                DataInstance(
                    inputs=api_seq,
                    outputs=doc,
                    split=split,
                    idx=str(js["id"])
                )
            )
            sizes[split] += 1
    assert sum(sizes.values()) == len(instances)
    sizes["total"] = len(instances)
    return instances, sizes


def read_tl_api_seq_sum_large(data_dir):
    data_dir = os.path.join(data_dir, "api_sum")

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
            api_seq = js["api_seq"]
            doc = js["comment"].strip()
            instances.append(
                DataInstance(
                    inputs=api_seq,
                    outputs=doc,
                    split=split,
                    idx=str(js["id"])
                )
            )
            sizes[split] += 1
    assert sum(sizes.values()) == len(instances)
    sizes["total"] = len(instances)
    return instances, sizes


def read_code_docstring_corpus_sum(data_dir):
    def escape_code(s):
        s = s.strip()
        # s = s.replace(" DCNL DCSP ", "\n")
        s = s.replace(" DCNL DCSP ", "\n\t")
        s = s.replace(" DCNL  DCSP ", "\n\t")
        s = s.replace(" DCNL ", "\n")
        s = s.replace(" DCSP ", " ")
        while "\t " in s:
            s = s.replace("\t ", "\t\t")
        return s

    def escape_nl(s):
        s = s.strip("'")
        s = s.strip()
        if "DCNL" in s:
            s = s.split("DCNL")[0]
        s = s.replace("DCSP", " ")
        return " ".join(s.split())

    data_dir = os.path.join(data_dir, "code_docstring_corpus")

    instances = []
    sizes = {
        "train": 0,
        "valid": 0,
        "test": 0
    }

    for split in ["train", "valid", "test"]:
        with open(os.path.join(data_dir, f"data_ps.declbodies.{split}"), mode="r", encoding="utf-8") as src_f, \
             open(os.path.join(data_dir, f"data_ps.descriptions.{split}"), mode="r", encoding="utf-8") as tgt_f, \
             open(os.path.join(data_dir, f"data_ps.metadata.{split}"), mode="r", encoding="utf-8") as meta_f:
            sources = src_f.readlines()
            targets = tgt_f.readlines()
            metas = meta_f.readlines()
        assert len(sources) == len(targets) == len(metas)
        for source, target, meta in tqdm(zip(sources, targets, metas), desc="Reading", total=len(sources)):
            source = escape_code(source)
            target = escape_nl(target)
            idx = "#L".join(meta.strip().split())
            instances.append(
                DataInstance(
                    inputs=source.strip(),
                    outputs=target.strip(),
                    split=split,
                    idx=idx
                )
            )
            sizes[split] += 1
    assert sum(sizes.values()) == len(instances)
    sizes["total"] = len(instances)
    return instances, sizes


def read_code_docstring_corpus_gen(data_dir):
    def escape_code(s):
        s = s.strip()
        # s = s.replace(" DCNL DCSP ", "\n")
        s = s.replace(" DCNL DCSP ", "\n\t")
        s = s.replace(" DCNL  DCSP ", "\n\t")
        s = s.replace(" DCNL ", "\n")
        s = s.replace(" DCSP ", " ")
        while "\t " in s:
            s = s.replace("\t ", "\t\t")
        return s

    def escape_body(s):
        s = escape_code(s)
        return s.replace("DCSP ", "\t")

    def escape_nl(s):
        s = s.strip("'")
        s = s.strip()
        if "DCNL" in s:
            s = s.split("DCNL")[0]
        s = s.replace("DCSP", " ")
        return " ".join(s.split())

    data_dir = os.path.join(data_dir, "code_docstring_corpus")

    instances = []
    sizes = {
        "train": 0,
        "valid": 0,
        "test": 0
    }

    for split in ["train", "valid", "test"]:
        with open(os.path.join(data_dir, f"data_ps.declarations.{split}"), mode="r", encoding="utf-8") as decl_f, \
             open(os.path.join(data_dir, f"data_ps.descriptions.{split}"), mode="r", encoding="utf-8") as desc_f, \
             open(os.path.join(data_dir, f"data_ps.bodies.{split}"), mode="r", encoding="utf-8") as body_f, \
             open(os.path.join(data_dir, f"data_ps.metadata.{split}"), mode="r", encoding="utf-8") as meta_f:
            decls = decl_f.readlines()
            descs = desc_f.readlines()
            targets = body_f.readlines()
            metas = meta_f.readlines()
        assert len(decls) == len(descs) == len(targets) == len(metas)

        for decl, desc, target, meta in zip(decls, descs, targets, metas):
            decl = escape_code(decl)
            desc = escape_nl(desc)
            target = escape_body(target)
            idx = "#L".join(meta.strip().split())
            instances.append(
                DataInstance(
                    inputs=[decl.strip(), desc.strip()],
                    outputs=target.strip(),
                    split=split,
                    idx=idx
                )
            )
            sizes[split] += 1
    assert sum(sizes.values()) == len(instances)
    sizes["total"] = len(instances)
    return instances, sizes


def read_concode(data_dir):

    data_dir = os.path.join(data_dir, "concode")

    instances = []
    sizes = {
        "train": 0,
        "valid": 0,
        "test": 0
    }
    for split in ["train", "valid", "test"]:
        with open(os.path.join(data_dir, f"{split}.jsonl"), mode="r", encoding="utf-8") as f:
            lines = f.readlines()
        for idx, line in enumerate(tqdm(lines, total=len(lines), desc=f"Loading {split} data")):
            js = json.loads(line.strip())
            source = " ".join(js["nl"])
            target = " ".join(js["code"])
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
    sizes["total"] = len(instances)
    return instances, sizes


def read_many_types_4_typescript(data_dir):

    def convert_target(tokens: list[str], labels: list[str]):
        assert len(tokens) == len(labels)
        results = [f"{token}: {label}" for token, label in zip(tokens, labels) if label is not None and label != "null"]
        if len(results) == 0:
            results.append("None")
        return results

    data_dir = os.path.join(data_dir, "many_types_4_typescript")
    instances = []
    sizes = {
        "train": 0,
        "valid": 0,
        "test": 0
    }
    for split in ["train", "valid", "test"]:
        with open(os.path.join(data_dir, f"{split}.jsonl"), mode="r", encoding="utf-8") as f:
            lines = f.readlines()
        for idx, line in enumerate(tqdm(lines, total=len(lines), desc=f"Loading {split} data")):
            js = json.loads(line.strip())
            source = " ".join(js["tokens"])
            target = convert_target(tokens=js["tokens"], labels=js["labels"])
            idx = "#".join([js["url"], js["path"], js["commit_hash"]])
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
    sizes["total"] = len(instances)
    return instances, sizes
