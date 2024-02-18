import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
import io
import numpy as np
import matplotlib.pyplot as plt
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
import weightwatcher as ww
import random


def get_model(args):
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf", token="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"
    )

    if args.freeze:
        for param in model.parameters():
            param.requires_grad = False
        return model
    
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
    
    return model

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
