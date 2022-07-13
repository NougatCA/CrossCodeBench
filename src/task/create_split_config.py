import json
import os

from utils import walk_tasks


def category_task(task_dir, task_name):
    with open(os.path.join(task_dir, f"{task_name}.meta.json"), mode="r", encoding="utf-8") as f:
        meta = json.load(f)
    if meta["Type"][0] == "Translation":
        return "eval"
    else:
        return "tune"


def write_config(task_dir, splits_to_tasks, config_name):
    split_dir = os.path.join(task_dir, "split")
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)
    with open(os.path.join(split_dir, f"{config_name}.json"), mode="w", encoding="utf-8") as f:
        json.dump(splits_to_tasks, f, indent=4)


def main():
    task_dir = "../../tasks/"
    config_name = "translation"
    # split to task name
    splits_to_tasks = {
        "tune": [],
        "eval": []
    }
    for task_name in walk_tasks(task_dir):
        split = category_task(task_dir, task_name)
        assert split in ["tune", "eval"]
        splits_to_tasks[split].append(task_name)

    write_config(task_dir, splits_to_tasks, config_name)


if __name__ == "__main__":
    main()
