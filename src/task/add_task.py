import json
import re

from utils import *


def create_meta_data(instances, sizes):

    meta = {
        # "Prompt Name": [
        #
        # ],
        # "Prompt id": [
        #
        # ],
        "Contributors": [
            "Default"
        ],
        "Source": [
            "wrong_binary_operator"
        ],
        "Type": [
            "Classification",
            "Binary"
        ],
        "BibTex": [
            """@inproceedings{kanade2020learning,
  title={Learning and evaluating contextual embedding of source code},
  author={Kanade, Aditya and Maniatis, Petros and Balakrishnan, Gogul and Shi, Kensen},
  booktitle={International Conference on Machine Learning},
  pages={5110--5121},
  year={2020},
  organization={PMLR}
}"""
        ],
        "URL": [
            "https://console.cloud.google.com/storage/browser/cubert/20200621_Python/wrong_binary_operator_datasets"
        ],
        "Categories": [
            "Detection -> Defect Detection"
        ],
        "Reasoning": [
            "Reasoning on code defect"
        ],
        "Definition": [
            "In this task, you are given a function, "
            "answer 'Wrong binary operator' if any binary operator in the function is incorrect, 'Correct' otherwise."
        ],
        "Input_language": [
            "Programming Language -> Python"
        ],
        "Output_language": [
            "Natural Language -> English"
        ],
        "Instruction_language": [
            "Natural Language -> English"
        ],
        "Domains": [
            "Operator"
        ],
        "Instance_number": [
            sizes
        ],
        "Positive Examples": [

        ],
        "Negative Examples": [

        ],

    }

    # if len(instances) > 65000:
    #     instances = random.sample(instances, 65000)

    data = {"Instances": []}
    for instance in instances:
        data["Instances"].append({
            "input": instance.inputs,
            "output": instance.outputs,
            "split": instance.split,
            "idx": instance.idx
        })
    return meta, data


def write_task(meta, data, task_dir):

    max_task_id = 0
    for file_name in os.listdir(task_dir):
        file_path = os.path.join(task_dir, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".json"):
            m = re.match(r"task_(\d\d\d)_.*", file_name)
            task_id = int(m.group(1))
            if task_id > max_task_id:
                max_task_id = task_id

    source = meta["Source"][0].lower()
    category = meta["Type"][0].lower()

    meta_filename = f"task_{str(max_task_id + 1).zfill(3)}_{source}_{category}.meta.json"
    meta_path = os.path.join(task_dir, meta_filename)
    data_filename = f"task_{str(max_task_id + 1).zfill(3)}_{source}_{category}.data.json"
    data_path = os.path.join(task_dir, data_filename)

    if os.path.exists(meta_path):
        raise ValueError(f"Meta file {meta_path} already exists.")
    if os.path.exists(data_path):
        raise ValueError(f"Data file {data_path} already exists.")

    with open(meta_path, mode="w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)
    print(f"Meta data dumped to {meta_path}")

    with open(data_path, mode="w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    num_instance = meta["Instance_number"][0]["total"]
    print(f"{num_instance} instances dumped as {data_path}")


def main():

    task_dir = "../../tasks/"
    data_dir = "../../datasets/"

    instances, sizes = read_wrong_binary_operator_dataset(data_dir)
    meta, data = create_meta_data(instances, sizes)
    write_task(meta, data, task_dir)


if __name__ == "__main__":
    main()
