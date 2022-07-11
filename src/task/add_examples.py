import os
import re
import json
from typing import Union, List
import random


class InputOutputPair(object):

    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.input_len = get_len(inputs)
        self.output_len = get_len(outputs)
        self.total_len = self.input_len + self.output_len


class Example(object):

    def __init__(self, inputs, outputs, reason=None):
        self.input = inputs
        self.output = outputs
        self.reason = reason


def get_len(item: Union[str, List[str]]):
    assert isinstance(item, str) or isinstance(item, list)
    if isinstance(item, str):
        return len(item)
    else:
        return sum([len(i) for i in item])


def random_sample_examples(data_file_path):
    with open(data_file_path, mode="r", encoding="utf-8") as f:
        data = json.load(f)
    instances = data["Instances"]
    pairs = []
    answers = []
    if len(instances) > 5000:
        instances = random.sample(instances, 5000)
    for instance in instances:
        pair = InputOutputPair(inputs=instance["input"], outputs=instance["output"])
        pairs.append(pair)
        if pair.outputs not in answers:
            answers.append(pair.outputs)
    sorted_pairs = sorted(pairs, key=lambda item: item.total_len)
    all_examples = random.sample(sorted_pairs[:20], k=8)
    random.shuffle(all_examples)
    pos_examples = all_examples[:4]
    neg_examples = all_examples[4:]

    for neg_example in neg_examples:
        origin_answer = neg_example.outputs
        if len(answers) == 2:
            neg_example.outputs = answers[0] if origin_answer == answers[1] else answers[1]
        else:
            while neg_example.outputs == origin_answer:
                neg_example.outputs = random.choice(answers)

    pos_examples = [{"input": pair.inputs, "output": pair.outputs, "reason": ""} for pair in pos_examples]
    neg_examples = [{"input": pair.inputs, "output": pair.outputs, "reason": ""} for pair in neg_examples]

    return pos_examples, neg_examples


def main():

    task_dir = "../../tasks"
    for file_name in os.listdir(task_dir):
        file_path = os.path.join(task_dir, file_name)
        if file_name.endswith(".meta.json") and os.path.isfile(file_path):
            print(file_name)
            data_file_name = re.sub(r".meta.json$", ".data.json", file_name)
            data_file_path = os.path.join(task_dir, data_file_name)
            pos_examples, neg_examples = random_sample_examples(data_file_path)

            with open(file_path, mode="r", encoding="utf-8") as f:
                data = json.load(f)
            data["Positive Examples"] = pos_examples
            data["Negative Examples"] = neg_examples
            with open(file_path, mode="w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()
