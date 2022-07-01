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
        # dataset, task name
        "Source": [
            "commit_gen"
        ],
        # Classification, Binary/Multi-label, Pairwise
        # Translation
        # Generation
        # Summarization
        # Tagging
        "Type": [
            "Tagging",
        ],
        "BibTex": [
            """@inproceedings{jiang2017automatically,
  title={Automatically generating commit messages from diffs using neural machine translation},
  author={Jiang, Siyuan and Armaly, Ameer and McMillan, Collin},
  booktitle={2017 32nd IEEE/ACM International Conference on Automated Software Engineering (ASE)},
  pages={135--146},
  organization={IEEE}
}""",
            """@inproceedings{liu2018neural,
  title={Neural-machine-translation-based commit message generation: how far are we?},
  author={Liu, Zhongxin and Xia, Xin and Hassan, Ahmed E and Lo, David and Xing, Zhenchang and Wang, Xinyu},
  booktitle={Proceedings of the 33rd ACM/IEEE International Conference on Automated Software Engineering},
  pages={373--384},
  year={2018}
}"""
        ],
        "URL": [
            "https://goo.gl/63B976",
        ],
        # Detection -> Defect/Clone Detection
        # Fill in the blank -> Exception Type
        # Classification -> Verification -> Docstring/Code Verification
        # Translation
        # Code Modification -> Bug Fixing
        # Named Entity Recognition -> Type Prediction
        # Summarization
        # Generation -> Commit Message
        "Categories": [
            "Generation -> Commit Message"
        ],
        # code defect
        # code semantic similarity
        # code semantic
        # code functionality
        # natural language and code semantic similarity
        # variable type
        "Reasoning": [
            "Reasoning on change diff"
        ],
        "Prompt": [
            "Generate commit message"
        ],
        "Definition": [
            "Given a change diff which captures the difference between two program versions, "
            "this task is to automatically generate a commit message, which is a textual description of the given diff, "
            "and can be also seen as the documentation of software changes."
        ],
        "Input_language": [
            "Programming Language -> Change Diff"
        ],
        "Output_language": [
            "Natural Language -> English"
        ],
        "Instruction_language": [
            "Natural Language -> English"
        ],
        # Software system security
        # Code semantic
        # Docstring
        # Variable
        # Operands
        # Operator
        # Bug
        # Code
        # Buffer
        # API
        # Commit message
        "Domains": [
            "Commit message"
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
    print(f"{num_instance} instances dumped to {data_path}")
    print("Split:")
    for k, v in meta["Instance_number"][0].items():
        print(f"{k}: {v}")


def main():

    task_dir = "../../tasks/"
    data_dir = "../../datasets/"

    instances, sizes = read_commit_gen(data_dir)
    meta, data = create_meta_data(instances, sizes)
    write_task(meta, data, task_dir)


if __name__ == "__main__":
    main()
