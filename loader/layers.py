import os
import torch
import random
import pandas as pd
from collections import defaultdict

def get_layers(args):    
    ww_details = pd.read_csv("llama_ww.csv")
    filtered = ww_details[  # type: ignore
        ww_details["longname"].str.contains("embed_tokens") == False  # type: ignore
    ]
    sortby = "random"
    if args.num_layers > len(filtered):
        args.num_layers = len(filtered)
    if "random" in (args.sortby).lower():
        train_names = random.sample(filtered['longname'].to_list(), args.num_layers)
    else:
        if "layer" in (args.sortby).lower():
            sortby = "layer_id"
        else:
            sortby = "alpha"
        train_names = (
            filtered.sort_values(by=[sortby], ascending=args.sort_ascending)[
                'longname'
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
    layer_to_train = list(set(layer_to_train))
    return layer_to_train

def save_layer_log(args, model, savepath):
    if not args.debug and "lora" not in args.sortby.lower():
        # Saving Details of Frozen Layers
        freeze_dict = None
        freeze_dict = defaultdict(list)
        for name, param in model.named_parameters():
            freeze_dict["name"].append(name)
            if param.grad is None:
                freeze_dict["freeze_layer"].append(True)
            elif torch.sum(param.grad.abs()).item() > 0:
                freeze_dict["freeze_layer"].append(False)
        if freeze_dict is not None:
            pd.DataFrame(freeze_dict).to_csv(
                os.path.join(savepath, f"freeze.csv")
            )

def param_count(m):
    params = sum([p.numel() for p in m.parameters()])/1_000_000
    trainable_params = sum([p.numel() for p in m.parameters() if p.requires_grad])/1_000_000
    print(f"Total params: {params:.2f}M, Trainable: {trainable_params:.2f}M")
    return params, trainable_params