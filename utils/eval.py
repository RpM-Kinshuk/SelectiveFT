import torch
import logging
from tqdm import tqdm
from utils.prompter import Prompter

# Validation Loss (Classification)
def eval_func(args, logger, trainer, all_metrics):
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)
    return all_metrics

def calc_acc(args, model, tokenizer, eval_dataset, disable_tqdm=True):
    correct = 0
    labels = []
    outputs = []
    batch_size = args.per_device_train_batch_size
    prompter = Prompter(args.prompt_template_name, verbose=False)
    with torch.no_grad():
        for start_idx in tqdm(range(0, len(eval_dataset), batch_size), disable=disable_tqdm):
            end_idx = min(start_idx + batch_size, len(eval_dataset))
            batch_inputs = eval_dataset['input'][start_idx:end_idx]
            batch_labels = eval_dataset['output'][start_idx:end_idx]
            inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True)
            input_ids = inputs['input_ids'].to(model.device) # type: ignore

            gen_output = model.generate(
                input_ids,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=5,
            )
            batch_outputs = gen_output.sequences
            # Decode outputs
            decoded_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
            for i, output in enumerate(decoded_outputs):
                output = prompter.get_response(output)
                labels.append(batch_labels[i])
                outputs.append(output)
    correct = sum([1 for i in range(len(labels)) if labels[i] in outputs[i]])
    return correct / len(labels)

def calc_val_loss(args, model, tokenizer, eval_dataloader, eval_dataset=None, disable_tqdm=True):
    """
    Args:
        model (_type_): _description_
        eval_dataloader (_type_): _description_
        device (_type_): _description_

    Returns:
        _type_: _description_
    """
    loss = 0
    val_acc = 0
    val_examples = 0
    model.eval()
    if eval_dataset is not None:
        val_acc = calc_acc(args, model, tokenizer, eval_dataset, disable_tqdm)
    with torch.no_grad():

        for step, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating", leave=False, disable=disable_tqdm)):
            input_len = len(batch["input_ids"])
            batch = {k: v.to(model.device) for k, v in batch.items()}
            output = model(**batch)
            loss += output.loss.item()
            val_examples += input_len
    return loss / len(eval_dataloader), val_acc