import pandas as pd
import random

def get_layers(args, layer_identifier='longname'):    
    ww_details = pd.read_csv("./llama_ww.csv")
    filtered = ww_details[  # type: ignore
        ww_details["longname"].str.contains("embed_tokens") == False  # type: ignore
    ]
    sortby = "random"
    if args.num_layers > len(filtered):
        args.num_layers = len(filtered)
    if "random" in (args.sortby).lower():
        train_names = random.sample(filtered["longname"].to_list(), args.num_layers)
    else:
        if "layer" in (args.sortby).lower():
            sortby = "layer_id"
        else:
            sortby = "alpha"
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
        if 'lora' in args.sortby.lower():
            layer_parts = layer.split(".")
            if len(layer_parts) > 2:
                layer_to_train.append(int(layer_parts[2]))
        else:
            layer_to_train.append(layer + ".weight")
    layer_to_train = list(set(layer_to_train))
    return layer_to_train