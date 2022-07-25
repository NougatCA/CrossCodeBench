from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, \
    T5ForConditionalGeneration, PLBartForConditionalGeneration, Seq2SeqTrainingArguments, \
    SchedulerType, IntervalStrategy
import torch
import logging
from tqdm import tqdm
import os
import numpy as np
import json
from prettytable import PrettyTable

from utils import human_format, count_params, layer_wise_parameters, Tuner, LogStateCallBack
from data import prepare_data
from metrics.exact_match import exact_match
from metrics.google_bleu import google_bleu
from metrics.smooth_bleu import smooth_bleu
from metrics.rouge import rouge_l


logger = logging.getLogger(__name__)


def run_tuning(args, run):
    logger.info("=" * 20 + " LOADING " + "=" * 20)

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

    # load model
    if args.random_init:
        if "t5" in args.model_name:
            model = T5ForConditionalGeneration(config)
        elif "plbart" in args.model_name:
            model = PLBartForConditionalGeneration(config)
        else:
            raise ValueError(f"Model name '{args.model_name}' not supported.")
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))
    logger.info(f"Loaded model '{model.__class__.__name__}' from '{args.model_name}'")
    logger.info(f"Trainable parameters: {human_format(count_params(model))}")
    logger.debug(f"Layer-wise parameters:\n{layer_wise_parameters(model)}")

    # prepare tuner
    tuning_args = Seq2SeqTrainingArguments(
        output_dir=args.model_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=False,
        do_predict=True,
        prediction_loss_only=False,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_epochs,
        lr_scheduler_type=SchedulerType.LINEAR,
        warmup_steps=args.warmup_steps,
        logging_dir=args.tb_dir,
        logging_strategy=IntervalStrategy.STEPS,
        logging_steps=100,
        save_strategy=IntervalStrategy.EPOCH,
        save_total_limit=5,
        seed=args.random_seed,
        bf16=args.mixed_precision == "bf16",
        fp16=args.mixed_precision == "fp16",
        dataloader_drop_last=False,
        run_name=args.short_run_name,
        ignore_data_skip=False,
        label_smoothing_factor=args.label_smoothing_factor,
        report_to=["tensorboard", "wandb"],
        predict_with_generate=True,
        dataloader_pin_memory=True)
    tuner = Tuner(
        model=model,
        args=tuning_args,
        tokenizer=tokenizer,
        data_collator=None,
        callbacks=[LogStateCallBack()])
    logger.info('Tuner is initialized')

    if not args.only_eval:
        tune_dataset, tune_dataloader = prepare_data(args, split="tune", tokenizer=tokenizer)
        tuner.tune_dataloader = tune_dataloader
        logger.info(f"Tuning data is loaded and prepared")

        logger.info("=" * 20 + " TUNING " + "=" * 20)
        tuner.train()
        logger.info("End of tuning")

    def decode_preds(preds):
        preds, labels = preds
        decoded_preds = tokenizer.decode_batch(preds)
        decoded_labels = tokenizer.decode_batch(labels)
        return decoded_labels, decoded_preds

    def compute_eval_metrics(eval_preds):
        decoded_labels, decoded_preds = decode_preds(eval_preds)
        results = {'references': decoded_labels, 'candidates': decoded_preds}
        all_golds = [ref.strip().split() for ref in decoded_labels]
        all_preds = [can.strip().split() for can in decoded_preds]
        results.update(exact_match(preds=all_preds, golds=all_golds))
        results.update(google_bleu(preds=all_preds, golds=all_golds))
        results.update(smooth_bleu(preds=all_preds, golds=all_golds))
        results.update(rouge_l(preds=all_preds, golds=all_golds))
        return results

    logger.info("=" * 20 + " EVALUATING " + "=" * 20)
    torch.cuda.empty_cache()

    # load eval data
    logger.info(f"Start loading evaluation data")
    eval_dataset, eval_dataloader = prepare_data(args, split="eval", tokenizer=tokenizer)
    tuner.eval_dataloader = eval_dataloader
    tuner.compute_metrics = compute_eval_metrics

    results = tuner.predict(metric_key_prefix='eval',
                            max_length=args.max_target_length,
                            num_beams=args.beam_width)

    metrics = results.metrics
    all_golds = metrics.pop("eval_golds")
    all_preds = metrics.pop("eval_preds")
    tuner.log_metrics(split='eval', metrics=metrics)
    tuner.save_metrics(split='eval', metrics=metrics)

    # # general statistics
    # num_examples = 0
    # num_steps = 0
    # loss_list = []
    # results = {}
    #
    # model.eval()
    #
    # all_preds = []
    # all_golds = []
    # for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Evaluating"):
    #     with torch.no_grad():
    #         input_ids = batch["input_ids"]
    #         attention_mask = batch["attention_mask"]
    #         if args.use_cuda:
    #             input_ids = input_ids.cuda()
    #             attention_mask = attention_mask.cuda()
    #         generated_tokens = model.generate(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             max_length=args.max_target_length,
    #             num_beams=args.num_beams,
    #             early_stopping=True
    #         )
    #
    #         generated_tokens = generated_tokens.cpu().numpy()
    #         labels = batch["labels"].cpu().numpy()
    #
    #         decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    #         decoded_golds = tokenizer.batch_decode(labels, skip_special_tokens=True)
    #
    #         all_preds.extend([pred.strip() for pred in decoded_preds])
    #         all_golds.extend([label.strip() for label in decoded_golds])
    #
    #         num_examples += batch["input_ids"].size(0)
    #         num_steps += 1
    #
    # # compute bleu, em, rouge-l, etc.
    # results.update(exact_match(preds=all_preds, golds=all_golds))
    # results.update(google_bleu(preds=all_preds, golds=all_golds))
    # results.update(smooth_bleu(preds=all_preds, golds=all_golds))
    # results.update(rouge_l(preds=all_preds, golds=all_golds))

    # save predictions and golds
    with open(os.path.join(args.eval_dir, "predictions.txt"), mode="w", encoding="utf-8") as pred_f, \
            open(os.path.join(args.eval_dir, "golds.txt"), mode="w", encoding="utf-8") as gold_f:
        for pred, gold in zip(all_preds, all_golds):
            pred_f.write(pred + "\n")
            gold_f.write(gold + "\n")

    # if len(loss_list) > 0:
    #     results.update({f"eval_loss": np.mean(loss_list)})
    # results.update({
    #     f"eval_num_examples": num_examples,
    #     f"eval_num_steps": num_steps
    # })

    # save results
    with open(os.path.join(args.eval_dir, "results.json"), mode="w", encoding="utf-8") as result_f:
        json.dump(results, result_f, indent=4)

    results_table = PrettyTable()
    results_table.field_names = ["Metric", "Score"]
    results_table.align["Metric"] = "c"
    results_table.align["Score"] = "l"
    for metric, score in results.items():
        results_table.add_row([metric, str(score)])

    logger.info(f"End of evaluating, results:\n{results_table}")
    run.log(results)
