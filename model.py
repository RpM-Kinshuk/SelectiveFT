import os
import io
import torch
import random
import numpy as np
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    pipeline,
    logging,
)
import weightwatcher as ww
from typing import Dict

import bitsandbytes as bnb # type: ignore
from peft import (
    prepare_model_for_kbit_training, # type: ignore
    LoraConfig, # type: ignore
    get_peft_model, # type: ignore
    PeftModel # type: ignore
)
from peft.tuners.lora import LoraLayer

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
if torch.cuda.is_available():   
    torch.backends.cuda.matmul.allow_tf32 = True # type: ignore

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
):
    """Borrowed from qlora codebase
    Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    
    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean( # type: ignore
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean( # type: ignore
            dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg # type: ignore
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg # type: ignore

def find_all_linear_names(args, model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')

    print(f'Found {len(lora_module_names)} linear layers')
    print(f'Linear layers: {lora_module_names}')
    return list(lora_module_names)

def get_model(args):

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    
    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        token="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",
        device_map=device_map,
        max_memory=max_memory,
        trust_remote_code=args.trust_remote_code,
    )

    model.config.use_cache = False
    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)
    model.config.torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        token="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",
        cache_dir=args.cache_dir,
        padding_side="right",
        use_fast=False, # Fast tokenizer giving issues.
        tokenizer_type='llama' if 'llama' in args.model_name_or_path else None, # Needed for HF name change
        trust_remote_code=args.trust_remote_code,
        # use_auth_token=args.use_auth_token,
    )
    if tokenizer._pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer, # type: ignore
            model=model,
        )
    if 'llama' in args.model_name_or_path or isinstance(tokenizer, LlamaTokenizer):
        # LLaMA tokenizer may not have correct special tokens set.
        # Check and add them if missing to prevent them from being parsed into different tokens.
        # Note that these are present in the vocabulary.
        # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
        print('Adding special tokens.')
        tokenizer.add_special_tokens({
                "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
                "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
                "unk_token": tokenizer.convert_ids_to_tokens(
                    model.config.pad_token_id
                    if model.config.pad_token_id != -1
                    else tokenizer.pad_token_id # type: ignore
                ),
        })
    
    if args.freeze and 'lora' not in args.sortby.lower():
        for name, param in model.named_parameters():
            param.requires_grad = False
            if "lm_head" in name:
                param.requires_grad = True
    elif 'lora' not in args.sortby.lower():
        for name, param in model.named_parameters():  # type: ignore
            param.requires_grad = True
        for name, module in model.named_modules():
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16) 
        return model, tokenizer

    # LORA INJECTION >>>-------------------------------------------->


    if 'lora' in args.sortby.lower():
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
        print(f'adding LoRA modules...')
        modules = find_all_linear_names(args, model)
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config) # type: ignore

    # SELECTIVE FINETUNING >>>------------------------------------->
    
    # if "lora" not in args.sortby.lower():
    #     # Save WeightWatcher Metrics
    #     watcher = ww.WeightWatcher(model=model)
    #     ww_details = watcher.analyze(min_evals=10)

    # if not args.debug and "lora" not in args.sortby.lower():
    #     ww_details.to_csv(os.path.join(stats_path, f"epoch_{epoch}.csv"))  # type: ignore

    ww_details = pd.read_csv("./llama_ww.csv")
    # CHOOSING LAYERS TO TRAIN BASED ON WEIGHTWATCHER METRICS/SORTBY
    if "lora" not in args.sortby.lower():
        filtered = ww_details[  # type: ignore
            ww_details["longname"].str.contains("embed_tokens") == False  # type: ignore
        ]
        sortby = "alpha"
        if args.num_layers > len(filtered):
            args.num_layers = len(filtered)
        if "random" in (args.sortby).lower():
            train_names = random.sample(filtered["longname"].to_list(), args.num_layers)
        else:
            if "alpha" in (args.sortby).lower():
                sortby = "alpha"
            elif "layer" in (args.sortby).lower():
                sortby = "layer_id"
            else:
                sortby = "random"
            train_names = (
                filtered.sort_values(by=[sortby], ascending=args.sort_ascending)[
                    "longname"
                ]
                .iloc[: args.num_layers]
                .to_list()
            )
        if args.verbose:
            print("Sorted by ", sortby)
            print("Training layers:", train_names)
        layer_to_train = []
        for layer in train_names:
            layer_to_train.append(layer + ".weight")
            layer_to_train.append(layer + ".bias")
            # Add Layer Norm
            if args.add_layer_norm:
                if "output" in layer:
                    layer_to_train.append(
                        layer.replace("dense", "LayerNorm") + ".weight"
                    )
                    layer_to_train.append(layer.replace("dense", "LayerNorm") + ".bias")
        layer_to_train = list(set(layer_to_train))
        # print("Final Training layers:", layer_to_train)
        for name, param in model.named_parameters():
            if name in layer_to_train:
                if args.verbose:
                    print(f"Enabling {name} parameter")
                param.requires_grad = True
    
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16) # type: ignore
        if 'norm' in name:
            module = module.to(torch.float32) # type: ignore
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16) # type: ignore
    
    return model, tokenizer

    # Randomly select layers to train without ww metrics
    param_optimizer = list(model.named_parameters())
    unique_prefixes = set()
    count = 0
    for name, _ in param_optimizer:
        # Extract the prefix by removing the last part (weight/bias)
        prefix = ".".join(name.split(".")[:-1])
        # Print only if the prefix is encountered for the first time
        if prefix not in unique_prefixes and "norm" not in prefix.lower():
            count += 1
            # print(prefix)
            unique_prefixes.add(prefix)
    print(count)
    count = 0
    random_layers = random.sample(unique_prefixes, args.num_layers)  # type: ignore
    print(f"Randomly selected layers: {random_layers}")
    return model
    uqd = set()
    for name, param in model.named_parameters():
        uqd.add(str(param.dtype))
    print(uqd)
    for name, param in model.named_parameters():
        if name.rsplit(".", 1)[0] in random_layers:
            print(name.rsplit(".", 1)[0])
            # if param.dtype in [torch.float32, torch.float64, torch.complex64, torch.complex128]:
            param.requires_grad = True
            count += 1
    print(f"Train layers: {count}\n\n")
    return model
