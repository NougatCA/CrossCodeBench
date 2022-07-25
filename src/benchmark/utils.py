from transformers import Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import torch

import logging
from prettytable import PrettyTable
import time

logger = logging.getLogger(__name__)


class Timer(object):
    """
    Computes elapsed time.
    """
    def __init__(self):
        self.running = True
        self.total = 0
        self.start = time.time()

    def reset(self):
        self.running = True
        self.total = 0
        self.start = time.time()
        return self

    def resume(self):
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    def time(self):
        if self.running:
            return self.total + time.time() - self.start
        return self.total


class Tuner(Seq2SeqTrainer):

    def __init__(self, tune_dataloader, **kwargs):
        super(Tuner, self).__init__(**kwargs)
        self.tune_dataloader = tune_dataloader

    def get_train_dataloader(self):
        return self.tune_dataloader


class LogStateCallBack(TrainerCallback):

    epoch_timer = Timer()

    def on_epoch_begin(self,
                       args: TrainingArguments,
                       state: TrainerState,
                       control: TrainerControl,
                       **kwargs):
        self.epoch_timer.reset()
        logger.debug('-' * 100)
        logger.debug(f'Start epoch {state.epoch}')

    def on_epoch_end(self,
                     args: TrainingArguments,
                     state: TrainerState,
                     control: TrainerControl,
                     optimizer: torch.optim.Optimizer,
                     **kwargs):
        epoch = state.epoch - 1
        self.map_step_epoch[state.global_step] = epoch
        logger.debug('Epoch {} / step {} finished, time: {:.2f}s'.format(epoch,
                                                                         state.global_step,
                                                                         self.epoch_timer.time()))
        logger.debug('learning rate: {}'.format(optimizer.param_groups[0]['lr']))


def get_run_name(args):
    short_name = get_short_run_name(args)
    tokens = [short_name, f"bs{args.train_batch_size}", f"ep{args.num_epochs}",
              f"lr{args.learning_rate}", f"warmup{args.num_warmup_steps}"]
    return "_".join([token for token in tokens if token is not None and token != ""])


def get_short_run_name(args):
    tokens = [args.init_model]
    if args.random_init:
        tokens.append("random")
    tokens.append(args.task_split_config)
    return "_".join([token for token in tokens if token is not None and token != ""])


def human_format(num):
    """Transfer count number."""
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'),
                         ['', 'K', 'M', 'B', 'T'][magnitude])


def count_params(model):
    """Count the number of learnable parameters of given model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def layer_wise_parameters(model):
    """Returns a printable table representing the layer-wise model parameters, their shapes and numbers"""
    table = PrettyTable()
    table.field_names = ["Layer Name", "Output Shape", "Param #"]
    table.align["Layer Name"] = "l"
    table.align["Output Shape"] = "r"
    table.align["Param #"] = "r"
    for name, parameters in model.named_parameters():
        if parameters.requires_grad:
            table.add_row([name, str(list(parameters.shape)), parameters.numel()])
    return table


def postprocess_results(result_dict: dict, major_metric=None):
    """
    Post-processes the evaluation result dict, such as generates result table and extracts major score.

    Args:
        result_dict (dict):
            A dict mapping from metric name to its score.
        major_metric (str, optional, defaults to None):
            The major metric name, if given, will return the major score.
    Returns:
        result_table (PrettyTable):
            A table of results.
        major_score (Union[None, float])
            The score corresponds to the major metric, `None` if `major_metric` is `None`.
    """
    results_table = PrettyTable()
    results_table.field_names = ["Metric", "Score"]
    results_table.align["Metric"] = "c"
    results_table.align["Score"] = "l"
    major_score = None
    for metric, score in result_dict.items():
        if major_metric and metric.endswith(major_metric):
            results_table.add_row([f"**{metric}**", score])
            major_score = score
        else:
            results_table.add_row([metric, str(score)])
    return results_table, major_score
