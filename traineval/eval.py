import torch
import logging
from tqdm import tqdm

# Validation Loss (Classification)
def eval_func(args, logger, trainer, all_metrics):
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)
    return all_metrics

def calc_val_loss(model, eval_dataloader, disable_tqdm=True):
    """
    Args:
        model (_type_): _description_
        eval_dataloader (_type_): _description_
        device (_type_): _description_

    Returns:
        _type_: _description_
    """
    loss = 0
    val_examples = 0
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating", leave=False, disable=disable_tqdm)):
            input_len = len(batch["input_ids"])
            batch = {k: v.to(model.device) for k, v in batch.items()}
            output = model(**batch)
            loss += output.loss.item()
            val_examples += input_len
    return loss / len(eval_dataloader), 0