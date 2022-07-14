
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, \
    T5ForConditionalGeneration, PLBartForConditionalGeneration

import logging
from prettytable import PrettyTable

logger = logging.getLogger(__name__)


def get_run_name(args):
    tokens = [args.model, args.task, args.dataset, args.subset,
              f"bs{args.train_batch_size}", f"ep{args.num_epochs}",
              f"lr{args.learning_rate}", f"warmup{args.num_warmup_steps}"]
    return "_".join([token for token in tokens if token is not None and token != ""])


def get_short_run_name(args):
    tokens = [args.model, args.task, args.dataset, args.subset]
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


def build_model_tokenizer(args) -> (PreTrainedModel, PreTrainedTokenizer):
    """Builds the model and tokenizer."""

    # load config
    config = AutoConfig.from_pretrained(args.model_name)
    logger.info(f"Loaded config '{config.__class__.__name__}' from '{args.model_name}'")
    logger.debug(config)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    args.pad_token_id = tokenizer.pad_token_id

    logger.info(f"Loaded tokenizer '{tokenizer.__class__.__name__}' from '{args.model_name}', "
                f"size: {len(tokenizer)}")
    # logger.debug(f"Special symbols: {tokenizer.all_special_tokens}")

    # load unwrapped model
    if args.random_init:
        if "t5" in args.model_name:
            model = T5ForConditionalGeneration(config)
        elif "plbart" in args.model_name:
            model = PLBartForConditionalGeneration(config)
        else:
            raise ValueError(f"Model name `{args.model_name}` not supported.")
    else:
        model = AutoModelForSeq2SeqLM(args.model_name)
    model.resize_token_embeddings(len(tokenizer))
    logger.info(f"Loaded model '{model.__class__.__name__}' from '{args.model_name}'")
    logger.info(f"Trainable parameters: {human_format(count_params(model))}")
    logger.debug(f"Layer-wise parameters:\n{layer_wise_parameters(model)}")

    return model, tokenizer
