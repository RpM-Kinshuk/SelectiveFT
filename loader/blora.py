import os
import random
import pandas as pd
from types import SimpleNamespace

def get_blocks(args):
    modif = 'None'
    if 'mid' in args.sortby:
        modif = 'mid'
    elif 'peak' in args.sortby:
        modif = 'peak'
    print(f"Using {args.model_name_or_path}/esd_{modif}.csv with {args.sortby}|Descending:{args.sort_ascending} layer-selection")
    ww_details = pd.read_csv(os.path.join('output', args.model_name_or_path, f"esd_{modif}.csv"))
    filtered = ww_details[  # type: ignore
        ww_details["longname"].str.contains("embed_tokens|lm_head") == False
    ]
    all_layers = filtered["longname"].to_list()
    num_blocks = int((all_layers[-1]).split(".")[2])+1
    tf_blocks = {}
    tf_alphas = {}

    for i in range(num_blocks):
        tf_blocks[i] = filtered[(filtered["longname"].str.split(".").str[2]) == f"{i}"]
        tf_alphas[i] = tf_blocks[i]["alpha"].sum()

    sortby = 1
    if 'layer' in args.sortby.lower():
        sortby = 0
    sorted_blocks = sorted(tf_alphas.items(), key=lambda x: x[sortby], reverse=not args.sort_ascending)
    if args.num_layers > num_blocks:
        args.num_layers = num_blocks
    if 'random' in args.sortby.lower():
        random.shuffle(sorted_blocks)
    # print(sorted_blocks)
    blocks_to_train = [x[0] for x in sorted_blocks[:args.num_layers]]
    blocks_to_train = list(set(blocks_to_train))
    return blocks_to_train