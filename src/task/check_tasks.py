
import os
import json


fields_to_check = [
    "Contributors",
    "Source",
    "Type",
    "BibTex",
    "URL",
    "Reasoning",
    "Categories",
    "Definition",
    "Input_language",
    "Output_language",
    "Instruction_language",
    "Domains",
    "Instance_number",
    "Positive Examples",
    "Negative Examples",
    "Instances"
]


def check_all_tasks():
    task_dir = "../../tasks/"
    for file_name in os.listdir(task_dir):
        file_path = os.path.join(task_dir, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".json"):
            with open(file_path, mode="r", encoding="utf-8") as f:
                data = json.load(f)
                for field in fields_to_check:
                    if field in data:
                        value = data[field]
                        if (isinstance(value, str) and value == "") or (isinstance(value, list) and len(value) == 0):
                            print(f"WARNING: {file_name}: empty field: '{field}'")
                    else:
                        print(f"ERROR: {file_name}: missing field: '{field}'.")


if __name__ == "__main__":
    check_all_tasks()
