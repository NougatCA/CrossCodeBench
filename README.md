# CrossCodeBench

## Requirements

Our basic experimental environment is Python 3.9.12, PyTorch 1.12.0 and CUDA 11.6.

The list of other requirements can be found in `requirements.txt`.

## Using Instructions

Run `main.py` to start experiments. All available arguments are located in `args.py`, specific whatever you need.

Some example scripts are as following.


2-shot PLBART on cat-intra-bf
```shell
python main.py \
--init_model plbart \
--use_few_shot \
--num_shots 2 \
--task_split_config cat-intra-bf
```

2/0-instruct CodeT5-large on sub-inter-c2t, and specific some parameters
```shell
python main.py \
--init_model codet5-large \
--use_instruction \
--num_neg_examples 0 \
--task_split_config sub-inter-c2t \
--train_batch_size 16 \
--eval_batch_size 8 \
--num_epochs 1 \
--learning_rate 1e-5 \
--warmup_steps 1000
```

If you need to run the supervised baselines, use the following scripts
```shell
python main.py \
--init_mdoel codet5 \
--supervised \
--task_split_config type-trans
```

## Artifacts

We provide the raw dataset, tasks, meta information and their summaries, which can be downloaded [here](https://doi.org/10.5281/zenodo.7321934).
Every directory is a dataset and can be parsed into several tasks.
The script to load most of the dataset can be found in `src/task/utils.py`, except ones that need to be pre-process by using other tools such as Java.

Each task corresponds to two Json files, `task_{id}_{name}.meta.json` and `task_{id}_{name}.data.json`.
The former contains the meta information of the task, while the latter consists of all data instances.

Extract the zip file and place the entire folder within the root directory.
