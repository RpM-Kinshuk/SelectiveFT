import os
import time
import torch
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from collections import defaultdict
from traineval.eval import calc_val_loss

def calc_train_loss(
    args, model, optimizer, device, train_dataloader, eval_dataloader, accelerator=None
):
    """
    Args:
        args: A dictionary of arguments
        model: A model object
        optimizer: An optimizer object
        device: A string of device name
        train_dataloader: A dataloader for training
        eval_dataloader: A dataloader for evaluation
        accelerator: A 🤗 Accelerator object

    Returns:
        train_losses: A list of training losses
        val_losses: A list of validation losses
        val_accs: A list of validation accuracies
    """
    model.train()
    num_all_pts = 0
    train_losses = []
    val_losses = []
    val_accs = []

    stats_path = os.path.join(args.savepath, "stats")
    Path(stats_path).mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    num_steps = args.epochs * len(train_dataloader)

    progress_bar = tqdm(
        range(num_steps),
        disable=not args.verbose,
    )

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        val_loss = 0
        tr_examples, tr_steps = 0, 0

        if args.verbose:
            print(f"===================================> Epoch {epoch+1}/{args.epochs}")
        # Training Loop
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            # TODO: UPDATE THIS TO WORK WITH LLAMA2
            outputs = model(
                # **batch,
                input_ids=batch["input_ids"].to(device),
                token_type_ids=batch["token_type_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            #######################################

            train_loss += outputs.loss.item()
            # if args.accelerate:
            #     accelerator.backward(outputs.loss)
            # else:
            outputs.loss.backward()
            optimizer.step()
            tr_examples += len(batch["labels"])
            num_all_pts += len(batch["labels"])
            tr_steps += 1
            train_losses.append(train_loss / tr_steps)

            if not args.debug and "lora" not in args.sortby.lower():
                # Saving Details of Frozen Layers
                freeze_dict = None
                if step in [0]:
                    freeze_dict = defaultdict(list)
                    for name, param in model.named_parameters():
                        freeze_dict["name"].append(name)
                        if param.grad is None:
                            freeze_dict["freeze_layer"].append(True)
                        elif torch.sum(param.grad.abs()).item() > 0:
                            freeze_dict["freeze_layer"].append(False)
                if freeze_dict is not None:
                    pd.DataFrame(freeze_dict).to_csv(
                        os.path.join(stats_path, f"freeze_{epoch}.csv")
                    )
            progress_bar.update(1)
            if args.task_name == 'wnli'and step >= 0.1 * len(train_dataloader):
                break
        time_elapsed = (time.time() - start_time) / 60

        # Validation Loss
        val_loss, val_acc = calc_val_loss(args, model, eval_dataloader, device)
        if args.verbose:
            print(
                f"\nEpoch: {epoch+1}/{args.epochs}"
                + f"|Elapsed: {time_elapsed:.2f} mins"
                + f"|Val Loss: {val_loss:.4f}|Val Acc: {val_acc:.4f}"
            )
        val_losses.append(val_loss)
        val_accs.append(val_acc)

    return train_losses, val_losses, val_accs