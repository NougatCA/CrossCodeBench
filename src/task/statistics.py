
import os
import json
import numpy as np


def update_or_create_count(key, dictionary):
    if key in dictionary:
        dictionary[key] += 1
    else:
        dictionary[key] = 1
    return dictionary


def print_dict(d):
    for key, value in d.items():
        print(f"{key}: {value}")


def main(root):

    total_sizes = []
    type_to_count = {}
    category_to_count = {}
    reason_to_count = {}
    input_lang_to_count = {}
    output_lang_to_count = {}
    lang_to_count = {}
    domain_to_count = {}

    if not os.path.exists(root) or os.path.isfile(root):
        raise ValueError(f"Path not exists or is not a directory: {root}")
    for file_name in os.listdir(root):
        if file_name.endswith(".meta.json"):
            print(file_name)
            with open(os.path.join(root, file_name), mode="r", encoding="utf-8") as f:
                data = json.load(f)
                # total size
                total_size = data["Instance_number"][0]["total"]
                total_sizes.append(total_size)
                # type
                task_type = data["Type"][0]
                type_to_count = update_or_create_count(task_type, type_to_count)
                # category
                category = data["Categories"][0]
                category_to_count = update_or_create_count(category, category_to_count)
                # reasoning on
                reason = data["Reasoning"][0]
                reason_to_count = update_or_create_count(reason, reason_to_count)
                # input and output language
                input_lang = data["Input_language"]
                output_lang = data["Output_language"]
                for in_lang in input_lang:
                    update_or_create_count(in_lang, input_lang_to_count)
                    update_or_create_count(in_lang, lang_to_count)
                for out_lang in output_lang:
                    update_or_create_count(out_lang, output_lang_to_count)
                    update_or_create_count(out_lang, lang_to_count)
                # domain
                domain = data["Domain"][0]
                domain_to_count = update_or_create_count(domain, domain_to_count)

    avg_total_size = np.mean(total_sizes)

    print("-" * 50)
    print(f"Avg. total size: {avg_total_size}")
    print("-" * 50)
    print("Type to count:")
    print_dict(type_to_count)
    print("-" * 50)
    print("Category to count:")
    print_dict(category_to_count)
    print("-" * 50)
    print("Reasoning type to count:")
    print_dict(reason_to_count)
    print("-" * 50)
    print("Input language to count:")
    print_dict(input_lang_to_count)
    print("-" * 50)
    print("Output language to count:")
    print_dict(output_lang_to_count)
    print("-" * 50)
    print("Language to count:")
    print_dict(lang_to_count)
    print("-" * 50)
    print("Domain to count:")
    print_dict(domain_to_count)


if __name__ == "__main__":
    tasks_root = "../../tasks/"
    main(tasks_root)
