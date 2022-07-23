
def exact_match(preds, golds, prefix=None) -> dict:
    assert len(preds) == len(golds)
    count = 0
    for pred, gold in zip(preds, golds):
        if pred == gold:
            count += 1
    avg_score = count / len(preds)
    return {f"{prefix}_em" if prefix else "em": avg_score * 100}
