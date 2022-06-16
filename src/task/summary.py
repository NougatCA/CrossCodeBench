
import os
import json
import pandas as pd

fields_to_summary = [
    "Contributors",
    "Source",
    "Type",
    "Categories",
    "Definition",
    "Input_language",
    "Output_language",
    "Instruction_language",
    "Domains",
    "Instance_number"
]


def main():

    task_dir = "../../tasks/"

    summary_list = []

    for file_name in os.listdir(task_dir):
        file_path = os.path.join(task_dir, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".meta.json"):
            with open(file_path, mode="r", encoding="utf-8") as f:
                data = json.load(f)

            summary = []
            for field in fields_to_summary:
                if field in data:
                    value = data[field]
                    if isinstance(value, list) and len(value) > 1:
                        summary.append("; ".join(value))
                    else:
                        if isinstance(value, list):
                            if len(value) == 0:
                                print(f"WARNING: {file_name}: empty field: '{field}'")
                            value = value[0]
                        try:
                            value = str(value)
                            summary.append(value)
                            if value == "":
                                print(f"WARNING: {file_name}: empty field: '{field}'")
                        except Exception:
                            summary.append("")
                            print(f"ERROR: {file_name}: field type cannot convert to str: '{field}: {value} ({type(value)})'")
                else:
                    summary.append("")
                    print(f"ERROR: {file_name}: missing field: '{field}'")

            summary_list.append(summary)

    df = pd.DataFrame(data=summary_list, columns=fields_to_summary)

    save_dir = os.path.join("../../tasks/", "summary")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    df.to_excel(os.path.join(save_dir, "summary.xlsx"))


if __name__ == "__main__":
    main()
