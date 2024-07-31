import os
cachedir = '/scratch/kinshuk/cache'
os.environ["HF_HOME"] = cachedir
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from tqdm.auto import tqdm
from model import get_model
from loader.layers import param_count, layer_log
from loader.data_module import make_data_module
from traineval.eval import eval_func, calc_val_loss
import json
import time
import torch
import random
import logging
import argparse
import numpy as np
import pandas as pd
import transformers
from pathlib import Path
# import accelerate.utils
import torch.backends.mps
import torch.backends.cudnn
from torch.cuda import (
    max_memory_allocated,
    reset_peak_memory_stats,
    reset_max_memory_allocated,
    memory_allocated,
)
from loader.logger import get_logger
from transformers import ( 
    set_seed,
    Seq2SeqTrainer,
)
from os.path import exists, join, isdir
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
from transformers.utils.logging import (
    set_verbosity_error as transformers_vb_err,
)
from datasets.utils.logging import (
    set_verbosity_error as datasets_vb_err,
)
from loader.esd_est import net_esd_estimator
from loader.layers import get_layers, layer_log
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf"
    )
    trust_remote_code: Optional[bool] = field(
        default=True,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={"help": "To use Huggingface auth token from Git Credentials."}
    )
    lora_modules: Optional[list[str]] = field(default_factory=list)

@dataclass
class DataArguments:
    eval_dataset_size: int = field(
        default=1024, metadata={"help": "Size of validation dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging, truncate the number of train examples."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging, truncate the number of eval examples."
        },
    )
    source_max_len: int = field(
        default=1024,
        metadata={"help": "Maximum source sequence length."},
    )
    target_max_len: int = field(
        default=256,
        metadata={"help": "Maximum target sequence length."},
    )
    dataset: str = field(
        default='alpaca',
        metadata={"help": "Which dataset to finetune on. See datamodule for options."}
    )
    dataset_format: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset format being used. [alpaca|chip2|self-instruct|hh-rlhf]"}
    )

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    seed: Optional[int] = field(
        default=7,
        metadata={"help": "Random seed for reproducibility."}
    )
    cache_dir: Optional[str] = field(
        default=cachedir,
    )
    verbose: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to print verbose output."}
    )
    memlog: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to log memory usage."}
    )
    freeze: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to freeze the model."}
    )
    sortby: str = field(
        default='random',
        metadata={"help": "Layer sorting method. [random|alpha|layer|LoRA|DoRA]"}
    )
    num_layers: int = field(
        default=0,
        metadata={"help": "Number of layers to train."}
    )
    sort_ascending: bool = field(
        default=False,
        metadata={"help": "Whether to train in ascending order of layer sorting method."}
    )
    eval_steps: Optional[int] = field(
        default=200,
        metadata={"help": "Frequency of evaluation per number of steps."}
    )
    add_layer_norm: bool = field(
        default=False,
        metadata={"help": "Whether to add layer norm to the layers being trained."}
    )
    train_on_source: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to train on the input in addition to the target text."}
    )
    mmlu_split: Optional[str] = field(
        default='eval',
        metadata={"help": "The MMLU split to run on"}
    )
    mmlu_dataset: Optional[str] = field(
        default='mmlu-fs',
        metadata={"help": "MMLU dataset to use: [`mmlu-zs`:zero-shot|`mmlu-fs`:few-shot]."}
    )
    do_mmlu_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run the MMLU evaluation."}
    )
    do_eval: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to run the evaluation."}
    )
    max_mmlu_samples: Optional[int] = field(
        default=None,
        metadata={"help": "If set, only evaluates on `max_mmlu_samples` of the MMMLU dataset."}
    )
    mmlu_source_max_len: int = field(
        default=2048,
        metadata={"help": "Maximum source sequence length for MMLU."}
    )
    adam8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit adam."}
    )
    double_quant: bool = field(
        default=False,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "Lora R dimension."}
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": " Lora alpha."}
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help":"Lora dropout."}
    )
    full_finetune: bool = field(
        default=False,
        metadata={"help": "Finetune the entire model without adapters."}
    )
    max_memory_MB: int = field(
        default=45000,
        metadata={"help": "Free memory per gpu."}
    )
    report_to: str = field(
        default='none',
        metadata={"help": "To use wandb or something else for reporting."}
    )
    output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})
    optim: str = field(default='paged_adamw_32bit', metadata={"help": 'The optimizer to be used'})
    per_device_train_batch_size: int = field(default=1, metadata={"help": 'The training batch size per GPU.'})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": 'Gradients to accumulate before performing an optimizer step'})
    max_steps: int = field(default=10000, metadata={"help": 'How many optimizer update steps to take'})
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'}) # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learnign rate'})
    remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=False, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    do_train: bool = field(default=True, metadata={"help": 'To train or not.'})
    lr_scheduler_type: str = field(default='constant', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    logging_steps: int = field(default=10, metadata={"help": 'The frequency of update steps after which to log the loss'})
    group_by_length: bool = field(default=True, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    save_strategy: str = field(default='steps', metadata={"help": 'When to save checkpoints'})
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})
    save_total_limit: int = field(default=40, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})

@dataclass
class GenerationArguments:
    # For more hyperparameters check:
    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    # Length arguments
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={"help": "Max number of new tokens to be generated in eval or prediction loops"
                          "if predict_with_generate is set."}
    )
    min_new_tokens : Optional[int] = field(
        default=None,
        metadata={"help": "Min number of new tokens to generate."}
    )

    # Generation strategy
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True)

    # Hyperparameters for logit manipulation
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0)
    repetition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)

gpus = torch.cuda.device_count()
memory_per_gpu = [0] * gpus
def memall(gpus=gpus):
    for i in range(gpus):
        memory_per_gpu[i] = torch.cuda.memory_allocated(i)
    return sum(memory_per_gpu)

def loss_fn(x, y):
    "A Flat CrossEntropy" 
    return torch.nn.functional.cross_entropy(x.view(-1, x.shape[-1]), y.view(-1))

def esd_logs(args, model, savepath, step=0):
    alphas = [None, 'xmin_peak', 'xmin_mid']
    alpha = None
    if 'peak' in args.sortby:
        alpha = alphas[1]
    elif 'mid' in args.sortby:
        alpha = alphas[2]
    label = alpha if alpha is not None else 'None'
    path = os.path.join(savepath, 'stats', label)
    Path(path).mkdir(parents=True, exist_ok=True)
    esd = net_esd_estimator(net=model, fix_fingers=alpha)
    esd = pd.DataFrame(esd)
    esd.to_csv(os.path.join(path, f"step_{step}.csv"))
    if False and step % 5000 == 0:
        layer_to_train = get_layers(args=args, predefined_ww=esd)
        for name, param in model.named_parameters():
            if name in layer_to_train:
                param.requires_grad = True
            else:
                param.requires_grad = False
        layer_log(args, model, path, step)

def train(args, training_args, model, tokenizer, train_dataloader, eval_dataloader, data_module, savepath):
    peek_memory = 0
    for device in range(gpus):
        reset_peak_memory_stats(device=device)
        reset_max_memory_allocated(device=device)
    epochs = 3 if args.dataset == 'oasst1' else 1
    
    times = []
    val_accs = []
    val_losses = []
    train_losses = []
    weight_memory = memall()
    total_time, forward_time, backward_time = 0, 0, 0
    input_memory, activation_memory, gradient_memory, optimizer_memory = 0, 0, 0, 0
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    optimizer.zero_grad()
    for epoch in range(epochs):
        tick = 0
        step = 0
        train_loss, tr_steps = 0, 0
        for _, batch in enumerate((train_dataloader)):
            
            model.train()
            tick = time.time()
            optimizer.zero_grad()
            curr = memall()
            batch = {k: v.to(model.device) for k, v in batch.items()}
            ipm = (memall() - curr)
            input_memory += ipm
            
            curr = memall()
            start = time.time()
            output = model(**batch)
            forward_time += time.time() - start
            activation_memory += (memall() - curr)
            
            start = time.time()
            # loss = loss_fn(out.logits, batch["labels"]) / args.gradient_accumulation_steps
            loss = output.loss
            loss.backward()
            backward_time += time.time() - start
            gradient_memory += (memall() - ipm - weight_memory - optimizer_memory)

            curr = memall()
            optimizer.step()
            if step == 0:
                optimizer_memory = (memall() - curr)
                if 'lora' not in args.sortby:
                    layer_log(args, model, savepath)
        
            loss = loss.cpu()
            train_loss += loss.item()
            tr_steps += 1
            train_losses.append(train_loss/tr_steps)
            total_time += time.time() - tick
            times.append(total_time)
            
            if step % 500 == 0:
                print(f'Seed:{args.seed} | {args.dataset} | {args.sortby}_{args.num_layers}_{args.sort_ascending} | Step: {step} | Train Loss: {train_loss/tr_steps}')
            torch.cuda.empty_cache()

            if step == args.max_steps or step % args.eval_steps == 0 or step == 0:
                model.eval()
                val_loss, val_acc = calc_val_loss(model, eval_dataloader)
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                print(f'Seed:{args.seed} | {args.dataset} | {args.sortby}_{args.num_layers}_{args.sort_ascending} | Step: {step} | Val Loss: {val_loss}')
                if step == args.max_steps:
                    break
            step += 1
        print(f'Epoch: {epoch} | Seed:{args.seed} | {args.dataset} | {args.sortby}_{args.num_layers}_{args.sort_ascending} | Train Loss: {train_loss/tr_steps}')

    total_memory = memall()
    peek_memory = sum([max_memory_allocated(i) for i in range(gpus)])

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

    base = {"train_loss": train_losses, "val_loss": val_losses, "val_acc": val_accs, "time": times}
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
        "activation_mem": (activation_memory / step) / 1e6,
        "grad_mem": (gradient_memory / step) / 1e6,
        "input_mem": (input_memory / step) / 1e6,
        "total_mem": total_memory / 1e6,
        "peak_mem": peek_memory / 1e6,
    }
    return base, all_metrics, memory_dict

def main():
    global logger
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments
    )) # type: ignore
    
    model_args, data_args, training_args, generation_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    
    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )
    print(args)
    
    # Control randomness
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    set_seed(args.seed)  # transformers seed

    asc = '_True' if args.sort_ascending else '_False'
    if args.verbose:
        task_info = (
            f"\n{args.dataset}\n"
            + f"Seed {args.seed}\n"
            + f"{args.sortby}{asc} fine-tuning\n"
            + f"{args.num_layers} Layers\n"
        )
        print(task_info)
    else:
        datasets_vb_err()
        transformers_vb_err()
        global _tqdm_active
        _tqdm_active = False
    
    if ('layer' not in args.sortby) and ('alpha' not in args.sortby):
        asc = ''
    if args.sortby == 'lora' or args.sortby == 'dora':
        args.num_layers = 0
    if 'full' in args.sortby:
        args.freeze = False
        args.num_layers = 250
        args.learning_rate = 2e-7
    savepath = f"{args.output_dir}/{args.model_name_or_path}/seed_{args.seed}/{args.dataset}/lr_{args.learning_rate}/batch_{args.per_device_train_batch_size}/{args.sortby}{asc}/layers_{args.num_layers}"
    Path(savepath).mkdir(parents=True, exist_ok=True)

    model, tokenizer = get_model(args)

    if 'full' in args.sortby:
        for param in model.parameters():
            param.requires_grad = True
    
    data_module = make_data_module(tokenizer=tokenizer, args=args) # type: ignore
    dataset = {k:v for k,v in data_module.items()}
    train_dataloader = DataLoader(
        dataset['train_dataset'], # type: ignore
        batch_size=args.per_device_train_batch_size,
        collate_fn=dataset['data_collator'],
        shuffle=True,
    )
    if args.verbose:
        print(train_dataloader.__len__())

    eval_dataloader = DataLoader(
        dataset['eval_dataset'], # type: ignore
        batch_size=args.per_device_train_batch_size,
        collate_fn=dataset['data_collator'],
        shuffle=False,
    )

    base, all_metrics, memory_dict = train(args, training_args, model, tokenizer, train_dataloader, eval_dataloader, dataset, savepath)

    memory_string = (
        f"Total param      : {memory_dict['total_param']}\n"
        f"Train param      : {memory_dict['train_param']}\n"
        f"Dataset          : {memory_dict['dataset']}\n"
        f"Method           : {memory_dict['method']}\n"
        f"Layers           : {memory_dict['layers']}\n"
        f"Batch size       : {memory_dict['batch_size']}\n"
        f"Learning Rate    : {memory_dict['lr']}\n"
        f"Eval Loss        : {memory_dict['eval_loss']}\n"
        f"Forward time     : {memory_dict['forward_time']} min\n"
        f"Backward time    : {memory_dict['backward_time']} min\n"
        f"Weight memory    : {memory_dict['weight_mem']} MB\n"
        f"Optimizer memory : {memory_dict['optimizer_mem']} MB\n"
        f"Activation memory: {memory_dict['activation_mem']} MB\n"
        f"Gradient memory  : {memory_dict['grad_mem']} MB\n"
        f"Input memory     : {memory_dict['input_mem']} MB\n"
        f"Total memory     : {memory_dict['total_mem']} MB\n"
        f"Peak memory      : {memory_dict['peak_mem']} MB\n"
    )
    if args.verbose: 
        print(memory_string)

    if args.memlog:
        Path(savepath).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(savepath, "finetune.npy"), base) # type: ignore
        with open(os.path.join(savepath, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))
        log_info = (
            f"\n\n{args.dataset} "
            + f"Seed {args.seed}"
            + f"Batch Size {args.per_device_train_batch_size} "
            + f"{args.sortby}{asc} fine-tuning "
            + f"{args.num_layers} Layers"
        )
        logger = get_logger(savepath, "memlog.log")
        logger.info(log_info)
        logger.info(f"\n{memory_string}\n")
        if (args.do_train or args.do_eval or args.do_predict):
            with open(os.path.join(savepath, "metrics.json"), "w") as fout:
                fout.write(json.dumps(all_metrics))

        with open(os.path.join(savepath,'stats.json'), 'w') as json_file:
            json.dump(memory_dict, json_file, indent=4)

if __name__ == "__main__":
    main()