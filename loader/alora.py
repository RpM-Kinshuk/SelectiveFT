import os
import torch
import bitsandbytes as bnb # type: ignore
from peft import (
    LoraConfig, # type: ignore
    get_peft_model, # type: ignore
    PeftModel # type: ignore
)
from peft.tuners.lora import LoraLayer

def alora_model(args, model, layers_to_train):
    print("Using ALORA")
    print("Layers to train: ", layers_to_train)
    adapter_list = []
    layer_modules = []
    modules = []
    for layer in layers_to_train:
        layer_parts = layer.split(".")
        if len(layer_parts) > 2:
            layer_index = int(layer_parts[2])
            modules = []
            modules.append(layer_parts[-2])
            if 'dora' in args.sortby.lower():
                modules.append('lora_magnitude_vector')
            layer_modules.append((layer_index, modules))
    for step, (layer_index, modules) in enumerate(layer_modules):
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=modules,
            layers_to_transform=layer_index,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            use_dora='dora' in args.sortby.lower(),
            use_rslora='rslora' in args.sortby.lower(),
        )
        if step == 0:
            model = get_peft_model(model, config, adapter_name="adapter_" + str(step)) # type: ignore
        else:
            model.add_adapter(peft_config=config, adapter_name= "adapter_" + str(step))
        adapter_list.append("adapter_" + str(step))
    model.base_model.set_adapter(adapter_list) # type: ignore
    return model