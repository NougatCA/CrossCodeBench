
import logging

from utils import build_model_tokenizer


logger = logging.getLogger(__name__)


def run_tuning(args, accelerator, run):
    logger.info("=" * 20 + " LOADING " + "=" * 20)

    model, tokenizer = build_model_tokenizer(args)

    # prepare data for training and validation
    if not args.only_test:
        pass
