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


def write_config(task_dir, splits_to_tasks, config_name):
    split_dir = os.path.join(task_dir, "split")
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)
    with open(os.path.join(split_dir, f"{config_name}.json"), mode="w", encoding="utf-8") as f:
        json.dump(splits_to_tasks, f, indent=4)


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


def main():
    task_dir = "../../tasks/"
    config_name = "qa"
    # split to task name
    splits_to_tasks = {
        "tune": [],
        "eval": []
    }
    for task_name in walk_tasks(task_dir):
        split = category_task_qa(task_dir, task_name)
        if split == "none":
            continue
        assert split in ["tune", "eval"]
        splits_to_tasks[split].append(task_name)

    write_config(task_dir, splits_to_tasks, config_name)


if __name__ == "__main__":
    main()
