import re


def token_acc(preds, golds, prefix=None) -> dict:

    def clean_token(token):
        return re.sub(r"\W", "", token.lower())

    assert len(preds) == len(golds)
    total_score = 0
    for pred, gold in zip(preds, golds):
        num_match = 0

        gold_tokens = gold.split()
        pred_tokens = pred.split()
        gold_len = len(gold_tokens)
        pred_len = len(pred_tokens)

        for idx in range(min(gold_len, pred_len)):
            gold_token = gold_tokens[idx]
            pred_token = pred_tokens[idx]
            if clean_token(gold_token) == clean_token(pred_token):
                num_match += 1

        if gold_len == 0:
            acc = 1 if pred_len == 0 else 0
        else:
            acc = num_match / gold_len
        total_score += acc

    avg_acc = total_score / len(preds)
    return {f"{prefix}_token_acc" if prefix else "token_acc": avg_acc * 100}
