
import argparse
import random
import os
import json
from tqdm import tqdm

from metrics.exact_match import exact_match
from metrics.google_bleu import google_bleu
from metrics.smooth_bleu import smooth_bleu
from metrics.rouge import rouge_l


def convert_input_output_to_str(s):
    return s if isinstance(s, str) else " ||| ".join(s)


def main():
    # parse arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.register("type", "bool", lambda v: v.lower() in ["yes", "true", "t", "1", "y"])

    parser.add_argument("--split", type=str, default="type-trans",
                        help="Task split config name.")
    parser.add_argument("--task_dir", type=str, default="../../tasks",
                        help="The directory where tasks store.")
    parser.add_argument("--max_eval_sample_per_task", type=int, default=500)
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed, 0 to disable.")
    args = parser.parse_args()

    if args.random_seed > 0:
        random.seed(args.random_seed)

    print("=" * 20 + " LOADING " + "=" * 20)
    # load split config
    split_config_path = os.path.join(args.task_dir, "split", f"{args.split}.json")
    print(f"Start loading task split configuration from '{split_config_path}'")
    with open(split_config_path, mode="r", encoding="utf-8") as f:
        split_config = json.load(f)
        task_names = split_config["eval"]

    print(f"Start loading task data from '{args.task_dir}'")

    all_golds = []
    all_output_preds = []
    all_input_preds = []

    p_bar = tqdm(task_names)
    for task_name in p_bar:
        meta_path = os.path.join(args.task_dir, f"{task_name}.meta.json")
        data_path = os.path.join(args.task_dir, f"{task_name}.data.json")

        if os.path.exists(meta_path) and os.path.exists(data_path):
            p_bar.set_description(task_name)

            # try to load meta and data from json file
            try:
                with open(os.path.join(args.task_dir, f"{task_name}.meta.json"),
                          mode="r", encoding="utf-8") as meta_f, \
                        open(os.path.join(args.task_dir, f"{task_name}.data.json"),
                             mode="r", encoding="utf-8") as data_f:
                    meta = json.load(meta_f)
                    data = json.load(data_f)
            except MemoryError:
                print(f"Run out of memory when loading data file of task '{task_name}', skipped.")
                continue

            # positive example outputs
            example_outputs = [convert_input_output_to_str(instance["output"]) for instance in meta["Positive Examples"]]

            # get the instances
            eval_instances = data["Instances"][:args.max_eval_sample_per_task]

            # generate preds
            output_preds = [random.choice(example_outputs) for _ in range(len(eval_instances))]
            input_preds = [convert_input_output_to_str(instance["input"]) for instance in eval_instances]
            golds = [convert_input_output_to_str(instance["output"]) for instance in eval_instances]

            all_golds += golds
            all_output_preds += output_preds
            all_input_preds += input_preds

        else:
            print(f"WARNING: Task '{task_name}' is skipped as it not presents "
                  f"in the given task directory: '{args.task_dir}'")

    assert len(all_golds) == len(all_output_preds) == len(all_input_preds)
    print(f"Data loaded, total size: {len(all_golds)}")

    print("=" * 20 + " EVALUATING " + "=" * 20)

    output_results = {}
    output_results.update(exact_match(preds=all_output_preds, golds=all_golds))
    output_results.update(google_bleu(preds=all_output_preds, golds=all_golds))
    output_results.update(smooth_bleu(preds=all_output_preds, golds=all_golds))
    output_results.update(rouge_l(preds=all_output_preds, golds=all_golds))

    print("Results of random selecting output of positive examples:")
    for key, value in output_results.items():
        print(f"{key}: {value}")

    input_results = {}
    input_results.update(exact_match(preds=all_input_preds, golds=all_golds))
    input_results.update(google_bleu(preds=all_input_preds, golds=all_golds))
    input_results.update(smooth_bleu(preds=all_input_preds, golds=all_golds))
    input_results.update(rouge_l(preds=all_input_preds, golds=all_golds))

    print("Results of copying input:")
    for key, value in input_results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
