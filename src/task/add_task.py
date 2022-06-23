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
            "code_trans_cs_java"
        ],
        # Classification, Binary/Multi-label, Pairwise
        # Translation
        "Type": [
            "Translation"
        ],
        "BibTex": [
            """@article{lu2021codexglue,
  title={Codexglue: A machine learning benchmark dataset for code understanding and generation},
  author={Lu, Shuai and Guo, Daya and Ren, Shuo and Huang, Junjie and Svyatkovskiy, Alexey and Blanco, Ambrosio and Clement, Colin and Drain, Dawn and Jiang, Daxin and Tang, Duyu and others},
  journal={arXiv preprint arXiv:2102.04664},
  year={2021}
}"""
        ],
        "URL": [
            "https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/code-to-code-trans",
        ],
        # Detection -> Defect/Clone Detection
        # Fill in the blank -> Exception Type
        # Classification -> Verification -> Docstring/Code Verification
        # Translation
        "Categories": [
            "Translation"
        ],
        # code defect
        # code semantic similarity
        # code semantic
        # code functionality
        # natural language and code semantic similarity
        "Reasoning": [

        ],
        "Definition": [
            "Given a code sequence in C#, generate an equivalent code sequence in Java with the same functionality."
        ],
        "Input_language": [
            "Programming Language -> C#"
        ],
        "Output_language": [
            "Programming Language -> Java"
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
        "Domains": [
            "Functionality"
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

    instances, sizes = read_code_trans_cs_java(data_dir)
    meta, data = create_meta_data(instances, sizes)
    write_task(meta, data, task_dir)


if __name__ == "__main__":
    main()
