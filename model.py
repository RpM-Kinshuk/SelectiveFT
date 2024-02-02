import torch
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
import random

def get_model(args):
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf", token="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"
    )
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
    random_layers = random.sample(unique_prefixes, args.num_layers) #type: ignore
    print(f"Randomly selected layers: {random_layers}")
    return model
    uqd = set()
    for name, param in model.named_parameters():
        uqd.add(str(param.dtype))
    print(uqd)
    for name, param in model.named_parameters():
        if name.rsplit('.', 1)[0] in random_layers:
            print(name.rsplit('.', 1)[0])
            # if param.dtype in [torch.float32, torch.float64, torch.complex64, torch.complex128]:
            param.requires_grad = True
            count += 1
    print(f"Train layers: {count}\n\n")
