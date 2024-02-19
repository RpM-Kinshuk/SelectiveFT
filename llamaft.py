#TODO: Handle extra args and save metrics systematically

from model import get_model
from traineval.eval import eval_func
from traineval.train import train_func
from loader.callbacks import mmlu_callback
from loader.data_module import make_data_module
import os
import json
import torch
import random
import logging
import argparse
import numpy as np
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
# from accelerate import Accelerator
from os.path import exists, join, isdir
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

from transformers.utils.logging import (
    set_verbosity_error as transformers_vb_err,
)
from datasets.utils.logging import (
    set_verbosity_error as datasets_vb_err,
)

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf"
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={"help": "To use Huggingface auth token from Git Credentials."}
    )

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
        default='/rscratch/tpang/kinshuk/cache',
    )
    verbose: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to print verbose output."}
    )
    memlog: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to log memory usage."}
    )
    freeze: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to freeze the model."}
    )
    sortby: str = field(
        default='random',
        metadata={"help": "Layer sorting method. [random|alpha|layer]"}
    )
    num_layers: int = field(
        default=0,
        metadata={"help": "Number of layers to train."}
    )
    sort_ascending: bool = field(
        default=False,
        metadata={"help": "Whether to train in ascending order of layer sorting method."}
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
    max_mmlu_samples: Optional[int] = field(
        default=None,
        metadata={"help": "If set, only evaluates on `max_mmlu_samples` of the MMMLU dataset."}
    )
    mmlu_source_max_len: int = field(
        default=2048,
        metadata={"help": "Maximum source sequence length for MMLU."}
    )
    full_finetune: bool = field(
        default=False,
        metadata={"help": "Finetune the entire model without adapters."}
    )
    max_memory_MB: int = field(
        default=12000,
        metadata={"help": "Free memory per gpu."}
    )
    report_to: str = field(
        default='none',
        metadata={"help": "To use wandb or something else for reporting."}
    )
    output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})
    optim: str = field(default='paged_adamw_32bit', metadata={"help": 'The optimizer to be used'})
    per_device_train_batch_size: int = field(default=1, metadata={"help": 'The training batch size per GPU.'})
    gradient_accumulation_steps: int = field(default=16, metadata={"help": 'Gradients to accumulate before performing an optimizer step'})
    max_steps: int = field(default=10000, metadata={"help": 'How many optimizer update steps to take'})
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'}) # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learnign rate'})
    remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=True, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
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


os.environ["TRANSFORMERS_CACHE"] = "/rscratch/tpang/kinshuk/cache"
os.environ["HF_DATASETS_CACHE"]="/rscratch/tpang/kinshuk/cache"

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
    os.environ["TRANSFORMERS_CACHE"] = args.cache_dir
    cuda_device = torch.cuda.current_device()

    # Memory Log Path
    mempath = (
        f"/rscratch/tpang/kinshuk/RpMKin/llama_ft/alpaca/"
        + f"trainseed_{args.seed}/{args.sortby}"
    )
    
    # Control randomness
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # accelerate.utils.set_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    set_seed(args.seed)  # transformers seed

    # Memory Stats Initialization
    reset_peak_memory_stats(device=cuda_device)
    reset_max_memory_allocated(device=cuda_device)
    start_memory = memory_allocated(device=cuda_device)

    if args.verbose:
        print("SEED:", args.seed)
        task_info = (
            f"\n\n\nDataset to finetune on: {args.dataset}\n\n\n"
            + f"alpha Decreasing: {not args.sort_ascending}\n\n\n"
            + f"Layers to train: {args.num_layers}\n\n\n"
            + f"Train randomly: {'random' in args.sortby.lower()}\n\n\n"
        )
        print(task_info)
    else:
        datasets_vb_err()
        transformers_vb_err()
        global _tqdm_active
        _tqdm_active = False

    # WIP >>>------------------------------------------>

    model, tokenizer = get_model(args)

    data_module = make_data_module(tokenizer=tokenizer, args=args) # type: ignore

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **{k:v for k,v in data_module.items() if k != 'predict_dataset'},
    )

    if args.do_mmlu_eval:
        trainer = mmlu_callback(args, tokenizer, trainer)

    all_metrics = {"run_name": args.run_name}

    # Train
    if args.do_train:
        all_metrics = train_func(args, logger, trainer, all_metrics)
    
    # Eval
    if args.do_eval:
        all_metrics = eval_func(args, logger, trainer, all_metrics)

    end_memory = memory_allocated(device=cuda_device)
    peek_memory = max_memory_allocated(device=cuda_device)
    print(
        f"\n\n\nMemory usage before: {start_memory} bytes\nMemory usage after: {int((end_memory/1024)/1024)}MB"
    )
    print(f"\nPeak Memory usage: {int((peek_memory/1024)/1024)}MB\n\n\n")

    # WIP <-----------------------------------------<<<

    if args.memlog: # Memory Logging
        log_info = (
            f"\n\n{args.dataset} "
            + f"{args.num_layers} Layers "
            + f"{args.sortby} "
            + f"ascending {args.alpha_ascending}"
        )
        Path(mempath).mkdir(parents=True, exist_ok=True)
        logger = get_logger(mempath, "memlog.log")
        logger.info(log_info)
        logger.info(
            f"\nMemory usage before: {(start_memory/1024)/1024}MB\n"
            + f"Memory usage after: {(end_memory/1024)/1024}MB"
        )
        logger.info(f"\nPeak Memory usage: {(peek_memory/1024)/1024}MB\n\n")

    if (args.do_train or args.do_eval or args.do_predict):
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))

if __name__ == "__main__":
    main()