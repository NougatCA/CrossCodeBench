import json
import os
from utils import walk_tasks
from category_to_task_type import convert_category_to_task_type


def category_task_exception(task_dir, task_name):
    with open(os.path.join(task_dir, f"{task_name}.meta.json"), mode="r", encoding="utf-8") as f:
        meta = json.load(f)
    if meta["Source"][0] == "exception_type":
        return "eval"
    else:
        return "tune"


def category_task_assert(task_dir, task_name):
    with open(os.path.join(task_dir, f"{task_name}.meta.json"), mode="r", encoding="utf-8") as f:
        meta = json.load(f)
    if meta["Categories"][0] == "Fill in the blank -> Assert Statement":
        return "eval"
    else:
        return "tune"


def category_task_api(task_dir, task_name):
    with open(os.path.join(task_dir, f"{task_name}.meta.json"), mode="r", encoding="utf-8") as f:
        meta = json.load(f)
    if meta["Categories"][0] == "Generation -> API Sequence":
        return "eval"
    else:
        return "tune"


def category_task_translation(task_dir, task_name):
    with open(os.path.join(task_dir, f"{task_name}.meta.json"), mode="r", encoding="utf-8") as f:
        meta = json.load(f)
    if meta["Type"][0] == "Translation":
        return "eval"
    else:
        return "tune"


def category_task_fixing(task_dir, task_name):
    with open(os.path.join(task_dir, f"{task_name}.meta.json"), mode="r", encoding="utf-8") as f:
        meta = json.load(f)
    if meta["Categories"][0] == "Code Modification -> Bug Fixing":
        return "eval"
    else:
        return "tune"


def category_task_multi_label(task_dir, task_name):
    with open(os.path.join(task_dir, f"{task_name}.meta.json"), mode="r", encoding="utf-8") as f:
        meta = json.load(f)
    if "Multi-label" in meta["Type"]:
        return "eval"
    else:
        return "tune"


def category_task_tagging(task_dir, task_name):
    with open(os.path.join(task_dir, f"{task_name}.meta.json"), mode="r", encoding="utf-8") as f:
        meta = json.load(f)
    if meta["Type"][0] == "Tagging":
        return "eval"
    else:
        return "tune"


def category_task_summarization(task_dir, task_name):
    with open(os.path.join(task_dir, f"{task_name}.meta.json"), mode="r", encoding="utf-8") as f:
        meta = json.load(f)
    if meta["Type"][0] == "Summarization":
        return "eval"
    else:
        return "tune"


def category_task_ruby(task_dir, task_name):
    with open(os.path.join(task_dir, f"{task_name}.meta.json"), mode="r", encoding="utf-8") as f:
        meta = json.load(f)
    all_lang = " ".join(meta["Input_language"] + meta["Output_language"])
    if "Ruby" in all_lang:
        return "eval"
    else:
        return "tune"


def category_task_intra_multi_label(task_dir, task_name):
    with open(os.path.join(task_dir, f"{task_name}.meta.json"), mode="r", encoding="utf-8") as f:
        meta = json.load(f)
    category = meta["Categories"][0]
    task_type = convert_category_to_task_type(category)
    if task_type == "Classification -> Multi-label":
        return "eval"
    elif task_type == "Classification -> Binary":
        return "tune"
    else:
        return "none"


def category_task_intra_code_to_text(task_dir, task_name):
    with open(os.path.join(task_dir, f"{task_name}.meta.json"), mode="r", encoding="utf-8") as f:
        meta = json.load(f)
    category = meta["Categories"][0]
    task_type = convert_category_to_task_type(category)
    if task_type.startswith("Generation"):
        if task_type == "Generation -> Code-to-Text":
            return "eval"
        else:
            return "tune"
    else:
        return "none"


def category_task_inter_multi_label(task_dir, task_name):
    with open(os.path.join(task_dir, f"{task_name}.meta.json"), mode="r", encoding="utf-8") as f:
        meta = json.load(f)
    category = meta["Categories"][0]
    task_type = convert_category_to_task_type(category)
    if task_type == "Classification -> Multi-label":
        return "eval"
    else:
        return "tune"


def category_task_qa(task_dir, task_name):
    with open(os.path.join(task_dir, f"{task_name}.meta.json"), mode="r", encoding="utf-8") as f:
        meta = json.load(f)
    category = meta["Categories"][0]
    task_type = convert_category_to_task_type(category)
    if task_type == "Question Answering":
        return "eval"
    else:
        return "tune"


def category_task_intra_fixing(task_dir, task_name):
    with open(os.path.join(task_dir, f"{task_name}.meta.json"), mode="r", encoding="utf-8") as f:
        meta = json.load(f)
    category = meta["Categories"][0]
    task_type = convert_category_to_task_type(category)
    if task_type.startswith("Generation"):
        if "Bug Fixing" in category:
            return "eval"
        else:
            return "tune"
    else:
        return "none"


def category_task_intra_assert(task_dir, task_name):
    with open(os.path.join(task_dir, f"{task_name}.meta.json"), mode="r", encoding="utf-8") as f:
        meta = json.load(f)
    category = meta["Categories"][0]
    task_type = convert_category_to_task_type(category)
    if task_type.startswith("Generation"):
        if meta["Categories"][0] == "Fill in the blank -> Assert Statement":
            return "eval"
        else:
            return "tune"
    else:
        return "none"


def category_task(meta, split_name):
    category = meta["Categories"][0]
    task_type = convert_category_to_task_type(category)
    if split_name == "intra_wrong_binary":
        if task_type.startswith("Classification"):
            if "Wrong Binary Operator" in category:
                return "eval"
            else:
                return "tune"
        else:
            return "none"
    elif split_name == "intra_fixing":
        if task_type.startswith("Generation"):
            if "Bug Fixing" in category:
                return "eval"
            else:
                return "tune"
        else:
            return "none"
    elif split_name == "intra_assert":
        if task_type.startswith("Generation"):
            if meta["Categories"][0] == "Fill in the blank -> Assert Statement":
                return "eval"
            else:
                return "tune"
        else:
            return "none"
    elif split_name == "intra_multi_label":
        if task_type == "Classification -> Multi-label":
            return "eval"
        elif task_type == "Classification -> Binary":
            return "tune"
        else:
            return "none"
    elif split_name == "inter_wrong_binary":
        if "Wrong Binary Operator" in category:
            return "eval"
        else:
            return "tune"
    elif split_name == "inter_fixing":
        if "Bug Fixing" in category:
            return "eval"
        else:
            return "tune"
    elif split_name == "inter_assert":
        if meta["Categories"][0] == "Fill in the blank -> Assert Statement":
            return "eval"
        else:
            return "tune"
    elif split_name == "inter_multi_label":
        if "Multi-label" in meta["Type"]:
            return "eval"
        else:
            return "tune"
    elif split_name == "type_translation":
        if task_type == "Translation":
            return "eval"
        else:
            return "tune"
    elif split_name == "type_qa":
        if task_type == "Question Answering":
            return "eval"
        else:
            return "tune"
    elif split_name == "intra_code2text":
        if task_type.startswith("Generation"):
            if task_type == "Generation -> Code-to-Text":
                return "eval"
            else:
                return "tune"
        else:
            return "none"
    elif split_name == "inter_code2text":
        if task_type == "Generation -> Code-to-Text":
            return "eval"
        else:
            return "tune"
    elif split_name == "cat-intra-cd":
        if task_type.startswith("Classification"):
            if category == "Detection -> Clone Detection":
                return "eval"
            else:
                return "tune"
        return "none"
    elif split_name == "cat-inter-cd":
        if category == "Detection -> Clone Detection":
            return "eval"
        else:
            return "tune"

    elif split_name == "rewrite-other-bf":
        if task_type.startswith("Generation"):
            if "Bug Fixing" in category:
                return "eval"
            elif task_type.startswith("Generation -> Rewrite"):
                return "none"
            else:
                return "tune"
        else:
            return "none"
    elif split_name == "type-bf":
        if task_type.startswith("Generation"):
            if "Bug Fixing" in category:
                return "eval"
            else:
                return "none"
        else:
            return "tune"
    else:
        raise ValueError(f"Split name {split_name} is not supported.")


def main():
    task_dir = "../../tasks/"
    config_name = "type-bf"
    # split to task name
    splits_to_tasks = {
        "tune": [],
        "eval": [],
        "sizes": {
            "tune": 0,
            "eval": 0
        }

    }
    for task_name in walk_tasks(task_dir):
        with open(os.path.join(task_dir, f"{task_name}.meta.json"), mode="r", encoding="utf-8") as f:
            meta = json.load(f)

        split = category_task(meta, split_name=config_name)

        if split == "none":
            continue
        assert split in ["tune", "eval"]
        splits_to_tasks[split].append(task_name)
        total_size = meta["Instance_number"][0]["total"]
        splits_to_tasks["sizes"][split] += total_size

    write_config(task_dir, splits_to_tasks, config_name)


def write_config(task_dir, splits_to_tasks, config_name):
    split_dir = os.path.join(task_dir, "split")
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)
    with open(os.path.join(split_dir, f"{config_name}.json"), mode="w", encoding="utf-8") as f:
        json.dump(splits_to_tasks, f, indent=4)


if __name__ == "__main__":
    main()
