
import re
import random

from utils import *


def create_meta_data(instances):

    data = {
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
            "bigclonebench"
        ],
        "BibTex": [
            """@inproceedings{svajlenko2014towards,
  title={Towards a big data curated benchmark of inter-project code clones},
  author={Svajlenko, Jeffrey and Islam, Judith F and Keivanloo, Iman and Roy, Chanchal K and Mia, Mohammad Mamun},
  booktitle={2014 IEEE International Conference on Software Maintenance and Evolution},
  pages={476--480},
  year={2014},
  organization={IEEE}
}""",
            """@inproceedings{wang2020detecting,
  title={Detecting Code Clones with Graph Neural Network and Flow-Augmented Abstract Syntax Tree},
  author={Wang, Wenhan and Li, Ge and Ma, Bo and Xia, Xin and Jin, Zhi},
  booktitle={2020 IEEE 27th International Conference on Software Analysis, Evolution and Reengineering (SANER)},
  pages={261--271},
  year={2020},
  organization={IEEE}
}"""
        ],
        "URL": [
            "https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Clone-detection-BigCloneBench"
        ],
        "Categories": [
            "Classification -> Binary"
        ],
        "Reasoning": [
            "Reasoning on code semantic"
        ],
        "Definition": [
            "You are given two pieces of code, your task is to identify whether they are semantically equivalent, "
            "Construct an answer that is 'True' if they are semantically equivalent and 'False' otherwise"
        ],
        "Input_language": [
            "Programming Language -> Java"
        ],
        "Output_language": [
            "Natural Language -> English"
        ],
        "Instruction_language": [
            "Natural Language -> English"
        ],
        "Domains": [
            "Code Clone"
        ],
        "Instance_number": [

        ],
        "Positive Examples": [

        ],
        "Negative Examples": [

        ],
        "Instances": [

        ]
    }

    if len(instances) > 65000:
        instances = random.sample(instances, 65000)

    for instance in instances:
        data["Instances"].append({
            "input": instance.inputs,
            "output": instance.outputs,
            "split": instance.split,
            "idx": instance.idx
        })
    return data


def write_task(data, task_dir):

    max_task_id = 0
    for file_name in os.listdir(task_dir):
        file_path = os.path.join(task_dir, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".json"):
            m = re.match(r"task_(\d\d\d)_.*", file_name)
            task_id = int(m.group(1))
            if task_id > max_task_id:
                max_task_id = task_id

    num_instance = len(data["Instances"])
    data["Instance_number"] = num_instance

    source = data["Source"][0]
    with open(os.path.join(task_dir, f"task_{str(max_task_id + 1).zfill(3)}_{source}_classification.json"), mode="w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"{num_instance} instances dumped.")


def main():
    task_dir = "../../tasks/"
    data_dir = os.path.join("../../datasets/", "bigclonebench")

    instances = read_bigclonebench_dataset(data_dir)
    data = create_meta_data(instances)
    write_task(data, task_dir)


if __name__ == "__main__":
    main()
