
import os
import json
import numpy as np

from transformers import AutoTokenizer, PreTrainedTokenizer


def update_or_create_count(key, dictionary, count=1):
    if key in dictionary:
        dictionary[key] += count
    else:
        dictionary[key] = count
    return dictionary


def get_field_str(field):
    return field if isinstance(field, str) else " ||| ".join(field)


def get_len(field):
    field_str = get_field_str(field)
    return len(field_str.split())


def get_token_len(field, tokenizer: PreTrainedTokenizer):
    field_str = get_field_str(field)
    tokens = tokenizer.tokenize(field_str)
    return len(tokens)


def print_dict(d):
    for key, value in d.items():
        print(f"{key}: {value}")


def main(root):

    num_tasks = 0

    prompt_lens = []
    definition_lens = []

    prompt_token_lens = []
    definition_token_lens = []

    pos_input_lens = []
    pos_output_lens = []
    neg_input_lens = []
    neg_output_lens = []

    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")

    if not os.path.exists(root) or os.path.isfile(root):
        raise ValueError(f"Path not exists or is not a directory: {root}")
    for file_name in os.listdir(root):
        if file_name.endswith(".meta.json"):
            print(file_name)
            with open(os.path.join(root, file_name), mode="r", encoding="utf-8") as f:
                data = json.load(f)
                # description length
                prompt = data["Prompt"][0]
                prompt_len = get_len(prompt)
                prompt_lens.append(prompt_len)
                prompt_token_len = get_token_len(prompt, tokenizer)
                prompt_token_lens.append(prompt_token_len)

                # definition
                definition = data["Definition"][0]
                definition_len = get_len(definition)
                definition_lens.append(definition_len)
                definition_token_len = get_token_len(definition, tokenizer)
                definition_token_lens.append(definition_token_len)

                # example
                pos_examples = data["Positive Examples"]
                for ex in pos_examples:
                    input_len = get_token_len(ex["input"], tokenizer)
                    output_len = get_token_len(ex["output"], tokenizer)
                    pos_input_lens.append(input_len)
                    pos_output_lens.append(output_len)

                neg_examples = data["Negative Examples"]
                for ex in neg_examples:
                    input_len = get_token_len(ex["input"], tokenizer)
                    output_len = get_token_len(ex["output"], tokenizer)
                    neg_input_lens.append(input_len)
                    neg_output_lens.append(output_len)

    print("-" * 50)
    print(f"Avg. prompt length: {np.mean(prompt_lens)}")
    print(f"Median prompt length: {np.median(prompt_lens)}")
    print(f"Avg. prompt token length: {np.mean(prompt_token_lens)}")
    print("-" * 50)
    print(f"Avg. definition length: {np.mean(definition_lens)}")
    print(f"Median definition length: {np.median(definition_lens)}")
    print(f"Avg. definition token length: {np.mean(definition_token_lens)}")
    print("-" * 50)
    print(f"Avg pos input length: {np.mean(pos_input_lens)}")
    print(f"Avg pos output length: {np.mean(pos_output_lens)}")
    print(f"Avg neg input length: {np.mean(neg_input_lens)}")
    print(f"Avg neg output length: {np.mean(neg_output_lens)}")
    print("-" * 50)
    print("-" * 50)


if __name__ == "__main__":
    tasks_root = "../../tasks/"
    main(tasks_root)
