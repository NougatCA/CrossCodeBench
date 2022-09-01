# CrossCodeBench

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

## Raw Datasets

We provide the raw dataset collected to create our benchmark, which can be downloaded [here](https://1drv.ms/u/s!Aj4XBdlu8BS0gf9eThJIS0fFBas1kA?e=pW7wq8).
Every directory is a dataset and can be parsed into several tasks.
The script to load most of the dataset can be found in `src/task/utils.py`, except ones that need to be pre-process by using other tools such as Java.

## Tasks with Meta Information

All tasks with meta information can be downloaded [here](https://1drv.ms/u/s!Aj4XBdlu8BS0gf9dMWr2g7jLTnC1oA?e=FZZD3F).
Each task corresponds to two Json files, `task_{id}_{name}.meta.json` and `task_{id}_{name}.data.json`.
The former contains the meta information of the task, while the latter consists of all data instances.

Extract the zip file and place the entire folder within the root directory.
