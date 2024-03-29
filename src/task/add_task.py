import os.path
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
            "fix_eval_verdict_python"
        ],
        # Classification, Binary/Multi-label, Pairwise
        # Translation
        # Generation
        # Summarization
        # Tagging
        "Type": [
            "Generation"
        ],
        "BibTex": [
            """@article{haque2022fixeval,
  title={FixEval: Execution-based Evaluation of Program Fixes for Competitive Programming Problems},
  author={Haque, Md Mahim Anjum and Ahmad, Wasi Uddin and Lourentzou, Ismini and Brown, Chris},
  journal={arXiv preprint arXiv:2206.07796},
  year={2022}
}"""
        ],
        "URL": [
            "https://github.com/mahimanzum/fixeval",
        ],
        # Detection -> Defect/Clone Detection
        # Fill in the blank -> Exception Type
        # Classification -> Verification -> Docstring/Code Verification
        # Translation
        # Code Modification -> Bug Fixing
        # Named Entity Recognition -> Type Prediction
        # Summarization
        # Generation -> Commit Message
        # Generation -> Program Synthesis
        "Categories": [
            "Code Modification -> Bug Fixing"
        ],
        # code defect
        # code semantic similarity
        # code semantic
        # code functionality
        # natural language and code semantic similarity
        # variable type
        "Reasoning": [
            "Reasoning on code defect"
        ],
        "Prompt": [
            "Generate fixed code"
        ],
        "Definition": [
            "Given a wrong competitive programming submission written in Python and the associated verdict, your task is to generate "
            "the corresponding accepted submission. Note that a wrong submission is the submission which fails to "
            "pass all test cases, or exceeds the time and memory specifications, etc. "
            "And the verdict is the type of error to which the failed commit belongs, such as Wrong Answer, Time Limit Exceeded, etc. "
            "And a submission is accepted when "
            "it is a feasible solution that passes all test cases and satisfies specific time and memory complexity requirements."
        ],
        "Input_language": [
            "Programming Language -> Python",
            "Natural Language -> English"
        ],
        "Output_language": [
            "Programming Language -> Python"
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
            "Code defect"
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
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)
    else:
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

    instances, sizes = read_fix_eval_verdict(data_dir, lang="python")
    meta, data = create_meta_data(instances, sizes)
    print("Writing meta and data files")
    write_task(meta, data, task_dir)


if __name__ == "__main__":
    main()
