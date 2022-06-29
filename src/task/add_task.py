import re

from utils import *


def create_meta_data(instances, sizes, lang):

    def get_lang_names(lang):
        if lang == "C++":
            return "cpp", "C++"
        elif lang == "Java":
            return "java", "Java"
        elif lang == "Python":
            return "python", "Python"
        elif lang == "C#":
            return "cs", "C#"
        elif lang == "Javascript":
            return "js", "JavaScript"
        elif lang == "PHP":
            return "php", "PHP"
        elif lang == "C":
            return "c", "C"
        else:
            raise ValueError

    lang_short, lang_formal = get_lang_names(lang)

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
            f"xlcost_gen_function_{lang_short}"
        ],
        # Classification, Binary/Multi-label, Pairwise
        # Translation
        # Generation
        # Summarization
        # Tagging
        "Type": [
            "Generation",
        ],
        "BibTex": [
            """@article{zhu2022xlcost,
  title={XLCoST: A Benchmark Dataset for Cross-lingual Code Intelligence},
  author={Zhu, Ming and Jain, Aneesh and Suresh, Karthik and Ravindran, Roshan and Tipirneni, Sindhu and Reddy, Chandan K},
  journal={arXiv preprint arXiv:2206.08474},
  year={2022}
}"""
        ],
        "URL": [
            "https://github.com/reddy-lab-code-research/XLCoST",
        ],
        # Detection -> Defect/Clone Detection
        # Fill in the blank -> Exception Type
        # Classification -> Verification -> Docstring/Code Verification
        # Translation
        # Code Modification -> Bug Fixing
        # Named Entity Recognition -> Type Prediction
        # Summarization
        "Categories": [
            "Code Generation"
        ],
        # code defect
        # code semantic similarity
        # code semantic
        # code functionality
        # natural language and code semantic similarity
        # variable type
        "Reasoning": [
            "Reasoning on natural language functionality"
        ],
        "Prompt": [
            f"Generate {lang_formal}"
        ],
        "Definition": [
            f"Given a text input, this task focuses on generating its corresponding {lang_formal} function."
        ],
        "Input_language": [
            "Natural Language -> English"
        ],
        "Output_language": [
            f"Programming Language -> {lang_formal}"
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
        "Domains": [
            "Code"
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


def main(lang):

    task_dir = "../../tasks/"
    data_dir = "../../datasets/"

    instances, sizes = read_xlcost_gen(data_dir,
                                       source_lang=lang,
                                       mode="program")
    meta, data = create_meta_data(instances, sizes, lang=lang)
    write_task(meta, data, task_dir)


if __name__ == "__main__":
    for source_lang in ["C++", "Java", "Python", "C#", "Javascript", "PHP", "C"]:
        main(lang=source_lang)
