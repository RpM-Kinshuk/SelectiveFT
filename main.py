# cachedir = '/rscratch/tpang/kinshuk/cache'
cachedir = '/scratch/kinshuk/cache'
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["TRANSFORMERS_CACHE"] = cachedir
os.environ["HF_DATASETS_CACHE"]= cachedir
from model import get_model
from loader.layers import param_count, layer_log
from loader.data_module import make_data_module
import json
import time
import torch
import random
import logging
import argparse
import numpy as np
import transformers
from pathlib import Path
import torch.backends.mps
import torch.backends.cudnn
from torch.cuda import (
    max_memory_allocated,
    reset_peak_memory_stats,
    reset_max_memory_allocated,
    memory_allocated,
)
from loader.logger import get_logger
from transformers import set_seed
# from accelerate import Accelerator
from torch.utils.data import DataLoader
from os.path import exists, join, isdir
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

from transformers.utils.logging import (
    set_verbosity_error as transformers_vb_err,
)
from datasets.utils.logging import (
    set_verbosity_error as datasets_vb_err,
)
from transformers import Seq2SeqTrainer
from traineval.eval import eval_func, calc_val_loss
logger = logging.getLogger(__name__)
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
from llamaft import ModelArguments, DataArguments, TrainingArguments, GenerationArguments
import gpustat
# Setting up the arguments

AVAILABLE_GPUS = [7]

time.sleep(3600)
sd = 73

# model_name = 'meta-llama/Llama-2-7b-hf'
model_name = 'meta-llama/Meta-Llama-3-8B'

model_args = ModelArguments(
    model_name_or_path=model_name
)

data_args = DataArguments(
    max_eval_samples=1000,
    dataset="alpaca", # DATASET [alpaca|chip2|self-instruct|hh-rlhf|oasst1|longform]
)

training_args = TrainingArguments(
    seed = sd,
    output_dir=f"./results/{model_name}/seed_{sd}",
    data_seed=7,
    evaluation_strategy="steps",
    do_eval=True,
    eval_steps=200,
    freeze = True,

    learning_rate=2e-6,     # LEARNING RATE
    
    max_steps=22500,       # NUMBER OF STEPS

    sortby="block_random",        # CAN DO "alpha" or "lora" or "dora"

    num_layers=14,           # NUMBER OF LAYERS FOR FULL FINE-TUNING

    per_device_train_batch_size = 1, # BATCH-SIZE
    memlog=True,
)

generation_args = GenerationArguments()

# If you need to use GenerationConfig or similar for generation_args
training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))

# Combine arguments into a single Namespace object (if needed)
args = argparse.Namespace(**vars(model_args), **vars(data_args), **vars(training_args))
args.cache_dir=cachedir

gs = 0
while(True):
    gs += 1
    flag = False
    stats = gpustat.GPUStatCollection.new_query()

    for i, stat in enumerate(stats.gpus):
        memory_used = stat['memory.used']
        if memory_used < 500 and i in AVAILABLE_GPUS:
            flag = True
            break
    if flag:
        break
    if gs == 1 or gs % 50 == 0:
        print(f"{args.sortby} Waiting for GPU to be free")
    time.sleep(10)

# Control randomness
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
set_seed(args.seed)  # transformers seed

if args.sortby == 'lora' or args.sortby == 'dora':
    args.num_layers = 0
if args.sortby == 'full':
    args.num_layers = 250
    args.learning_rate = 2e-7
logger = logging.getLogger(__name__)
gpus = torch.cuda.device_count()
start_memory = [0] * gpus
end_memory = [0] * gpus
peek_memory = 0

if not args.verbose:
    datasets_vb_err()
    transformers_vb_err()
    global _tqdm_active
    _tqdm_active = False

def memall(gpus=gpus):
    for i in range(gpus):
        start_memory[i] = torch.cuda.memory_allocated(i)
    return sum(start_memory)

savepath = f"./results/{args.model_name_or_path}/seed_{args.seed}/{args.dataset}/lr_{args.learning_rate}/batch_{args.per_device_train_batch_size}/{args.sortby}/layers_{args.num_layers}"
Path(savepath).mkdir(parents=True, exist_ok=True)

model, tokenizer = get_model(args)

if 'full' in args.sortby:
    for param in model.parameters():
        param.requires_grad = True

for device in range(gpus):
    reset_peak_memory_stats(device=device)
    reset_max_memory_allocated(device=device)

weight_memory = memall()

def loss_fn(x, y):
    "A Flat CrossEntropy" 
    return torch.nn.functional.cross_entropy(x.view(-1, x.shape[-1]), y.view(-1))

data_module = make_data_module(tokenizer=tokenizer, args=args) # type: ignore
dataset = {k:v for k,v in data_module.items()}
train_dataloader = DataLoader(
    dataset['train_dataset'], # type: ignore
    batch_size=args.per_device_train_batch_size,
    collate_fn=dataset['data_collator'],
    shuffle=True,
)

eval_dataloader = DataLoader(
    dataset['eval_dataset'], # type: ignore
    batch_size=args.per_device_train_batch_size,
    collate_fn=dataset['data_collator'],
    shuffle=False,
)

train_losses = []
val_losses = []
val_accs = []
times = []

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
optimizer.zero_grad()
optimizer_memory = 0
forward_time = 0
backward_time = 0

for epoch in range(1):
    train_loss = 0
    tr_steps = 0
    tick = 0
    total_time = 0
    for step, batch in enumerate((train_dataloader)):
        
        model.train()
        tick = time.time()
        optimizer.zero_grad()
        curr = memall()
        batch = {k: v.to(model.device) for k, v in batch.items()}
        input_memory = memall() - curr
        
        curr = memall()
        start = time.time()
        output = model(**batch)
        forward_time += time.time() - start
        activation_memory = memall() - curr
        
        curr = memall()
        start = time.time()
        # loss = loss_fn(out.logits, batch["labels"]) / args.gradient_accumulation_steps
        loss = output.loss
        loss.backward()
        backward_time += time.time() - start
        gradient_memory = memall() - input_memory - weight_memory - optimizer_memory

        curr = memall()
        optimizer.step()
        if step == 0:
            optimizer_memory = memall() - curr
            layer_log(args, model, savepath)
    
        loss = loss.cpu()
        train_loss += loss.item()
        tr_steps += 1
        train_losses.append(train_loss/tr_steps)
        total_time += time.time() - tick
        times.append(total_time)
        
        if step % 500 == 0:
            print(f'Seed: {args.seed}|{args.sortby}_{args.num_layers}|Step: {step}|Train Loss: {train_loss/tr_steps}')
        torch.cuda.empty_cache()

        if step % args.eval_steps == 0 or step == 0:
            model.eval()
            val_loss, val_acc = calc_val_loss(model, eval_dataloader)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            print(f'Seed: {args.seed}|{args.sortby}_{args.num_layers}|Step: {step}|Val Loss: {val_loss}')
            model.train()
        
        if step == args.max_steps:
            model.eval()
            val_loss, val_acc = calc_val_loss(model, eval_dataloader)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            print(f'Seed: {args.seed}|{args.sortby}_{args.num_layers}|Step: {step}|Val Loss: {val_loss}')
            break

total_memory = memall()

optimizer.zero_grad()
model.eval()
trainer=Seq2SeqTrainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                **{k:v for k,v in data_module.items() if k != 'predict_dataset'},
            )
all_metrics = {"run_name": args.run_name}
if args.do_eval:
    all_metrics = eval_func(args, logger, trainer, all_metrics)
total_memory = memall()
peek_memory = sum([max_memory_allocated(i) for i in range(gpus)])

memory_string = (
        f"{param_count(model)}\n"
        f"Dataset          : {args.dataset}\n"
        f"Method           : {args.sortby}\n"
        f"Layers           : {args.num_layers}\n"
        f"Batch size       : {args.per_device_train_batch_size}\n"
        f"Learning Rate    : {args.learning_rate}\n"
        f"Forward time     : {forward_time/60} min\n"
        f"Backward time    : {backward_time/60} min\n"
        f"Weight memory    : {weight_memory / 1e6} MB\n"
        f"Optimizer memory : {optimizer_memory / 1e6} MB\n"
        f"Activation memory: {activation_memory / 1e6} MB\n"
        f"Gradient memory  : {gradient_memory / 1e6} MB\n"
        f"Input memory     : {input_memory / 1e6} MB\n"
        f"Total memory     : {total_memory / 1e6} MB\n"
        f"Peak memory      : {peek_memory / 1e6} MB\n"
    )
print(memory_string)

# SAVE TRAINING HISTORY
base = {"train_loss": train_losses, "val_loss": val_losses, "val_acc": val_accs, "time": times}
if args.memlog:
    Path(savepath).mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(savepath, "finetune.npy"), base) # type: ignore
    with open(os.path.join(savepath, "metrics.json"), "w") as fout:
        fout.write(json.dumps(all_metrics))
    log_info = (
        f"\n\n{args.dataset} "
        + f"Batch Size {args.per_device_train_batch_size} "
        + f"{args.sortby} fine-tuning "
        + f"{args.num_layers} Layers"
    )
    logger = get_logger(savepath, "memlog.log")
    logger.info(log_info)
    logger.info(f"\n{memory_string}\n")
    if (args.do_train or args.do_eval or args.do_predict):
        with open(os.path.join(savepath, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))
    
    memory_dict = {
        "total_param": param_count(model)[0],
        "train_param": param_count(model)[1],
        "dataset": args.dataset,
        "method": args.sortby,
        "layers": args.num_layers,
        "batch_size": args.per_device_train_batch_size,
        "lr": args.learning_rate,
        "eval_loss": all_metrics["eval_loss"],
        "forward_time": forward_time / 60,
        "backward_time": backward_time / 60,
        "weight_mem": weight_memory / 1e6,
        "optimizer_mem": optimizer_memory / 1e6,
        "activation_mem": activation_memory / 1e6,
        "grad_mem": gradient_memory / 1e6,
        "input_mem": input_memory / 1e6,
        "total_mem": total_memory / 1e6,
        "peak_mem": peek_memory / 1e6,
    }
    with open(os.path.join(savepath,'stats.json'), 'w') as json_file:
        json.dump(memory_dict, json_file, indent=4)

exit(0)