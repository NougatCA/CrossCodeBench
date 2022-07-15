from accelerate import Accelerator
import torch
from torch.optim import AdamW
from transformers import get_scheduler
import logging
import math
from tqdm import tqdm

from utils import build_model_tokenizer, LabelSmoother
from data import prepare_data


logger = logging.getLogger(__name__)


def run_tuning(args, accelerator, run):
    logger.info("=" * 20 + " LOADING " + "=" * 20)

    model, tokenizer = build_model_tokenizer(args)

    # prepare data for training
    if not args.only_test:
        tune_dataset, tune_dataloader = prepare_data(args, split="tune", tokenizer=tokenizer)
        logger.info(f"Data is loaded and prepared")

        logger.info("=" * 20 + " TRAINING " + "=" * 20)
        # Optimizer
        # Split weights in two groups, one with weight decay and the other not
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

        # Prepare everything with accelerator
        model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, tune_dataloader)

        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_epochs * num_update_steps_per_epoch
        else:
            args.num_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        # Scheduler and math around the number of training steps
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )

        # Label smoothing
        if args.label_smoothing_factor != 0:
            label_smoother = LabelSmoother(epsilon=args.label_smoothing_factor)
        else:
            label_smoother = None

        total_batch_size = args.train_batch_size * args.num_gpus * args.gradient_accumulation_steps

        logger.info("***** Running tuning *****")
        logger.info(f"  Num examples = {len(tune_dataset)}")
        logger.info(f"  Num Epochs = {args.num_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")

        completed_steps = 0
        for epoch in range(args.num_epochs):
            model.train()

            train_bar = tqdm(train_dataloader, total=len(train_dataloader), desc=f"[epoch {epoch}, loss x.xxxx]")
            for step, batch in enumerate(train_bar):
                model_kwargs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[0].ne(args.pad_token_id),
                    "labels": batch[1],
                    "decoder_attention_mask": batch[1].ne(args.pad_token_id)
                }

                if label_smoother is not None:
                    labels = model_kwargs.pop("labels")
                else:
                    labels = None

                outputs = model(**model_kwargs)

                if labels is not None:
                    loss = label_smoother(outputs, labels)
                else:
                    loss = outputs.loss

                if args.num_gpus > 1:
                    loss = loss.mean()

                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
                if args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    completed_steps += 1

                    train_bar.set_description(f"[epoch {epoch}, loss {loss.item():.4f}]")
                    run.log({"train_loss": loss.item(), "epoch": epoch})

                if completed_steps >= args.max_train_steps:
                    break

        logger.info("End of tuning")

    logger.info("=" * 20 + " TESTING " + "=" * 20)
    torch.cuda.empty_cache()

    # load test data
    logger.info(f"Start loading test data")
    test_examples, test_dataset, test_dataloader = prepare_data(args, split="test", tokenizer=tokenizer)
    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    test_results = run_eval(args,
                            model=model,
                            tokenizer=tokenizer,
                            dataloader=test_dataloader,
                            accelerator=accelerator,
                            run=run,
                            raw_examples=test_examples,
                            split="test")
    result_table, _ = postprocess_results(test_results, major_metric=args.major_metric)
    logger.info(f"End of testing, results:\n{result_table}")
    run.log(test_results)
