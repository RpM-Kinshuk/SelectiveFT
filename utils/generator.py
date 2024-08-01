import os
import random
from datasets import Dataset, DatasetDict
from utils.prompter import Prompter

glue_task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

id_to_label = {
    "cola": {0: "unacceptable", 1: "acceptable"},
    "mnli": {0: "entailment", 1: "neutral", 2: "contradiction"},
    "mrpc": {0: "not_equivalent", 1: "equivalent"},
    "qnli": {0: "entailment", 1: "not_entailment"},
    "qqp": {0: "not_duplicate", 1: "duplicate"},
    "rte": {0: "entailment", 1: "not_entailment"},
    "sst2": {0: "negative", 1: "positive"},
    "stsb": {0: "low_similarity", 1: "high_similarity"}, # For simplicity, treating stsb as a binary task here
    "wnli": {0: "not_entailment", 1: "entailment"}
}

def glue_data(args, tokenizer, raw_dataset):

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    
    prompter = Prompter(args.prompt_template_name, verbose=False)
    task_to_keys = glue_task_to_keys

    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        sentence1_key, sentence2_key = "sentence1", "sentence2"
    
    labels_description = ", ".join([f"{v}" for k, v in id_to_label[args.task_name].items()])
    if ', ' in labels_description:
        labels_description = labels_description.rsplit(', ', 1)
        labels_description = ' or '.join(labels_description)

    train_dataset = raw_dataset["train"]
    eval_dataset = raw_dataset["validation_matched" if args.task_name == "mnli" else "validation"]

    def tokenize_prompt(data_point):
        instruction = "Classify the relationship between the following sentences:"
        input_text = f"{data_point[sentence1_key]}\n{data_point[sentence2_key]}" if sentence2_key else data_point[sentence1_key]
        output_text = id_to_label[args.task_name][data_point['label']]
        
        full_prompt = prompter.generate_prompt(instruction, input_text, output_text)
        tokenized_full_prompt = tokenizer(full_prompt, truncation=True, max_length=args.max_new_tokens, padding=False, return_tensors=None)
        
        if (
            tokenized_full_prompt["input_ids"][-1] != tokenizer.eos_token_id
            and len(tokenized_full_prompt["input_ids"]) < args.max_new_tokens
        ):
            tokenized_full_prompt["input_ids"].append(tokenizer.eos_token_id)
            tokenized_full_prompt["attention_mask"].append(1)
        tokenized_full_prompt["labels"] = tokenized_full_prompt["input_ids"].copy()

        if not args.train_on_source:
            user_prompt = prompter.generate_prompt(instruction, input_text)
            tokenized_user_prompt = tokenizer(user_prompt, truncation=True, max_length=args.max_new_tokens, padding=False, return_tensors=None)
            user_prompt_len = len(tokenized_user_prompt["input_ids"]) - 1
            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
    
    def create_prompt(data_point):
        instruction = f"Classify the relationship between the two sentences as {labels_description}."
        input = f"{sentence1_key}: {data_point[sentence1_key]}\n{sentence2_key}: {data_point[sentence2_key]}" if sentence2_key else data_point[sentence1_key]
        output = id_to_label[args.task_name][data_point['label']]
        text = prompter.generate_prompt(instruction, input, output)
        return {
            "instruction": instruction,
            "input": input,
            "output": output,
            "text": text
        }

    def create_dataset_dict(dataset):
        prompts = [create_prompt(data_point) for data_point in dataset]
        return Dataset.from_dict({
            "instruction": [p["instruction"] for p in prompts],
            "input": [p["input"] for p in prompts],
            "output": [p["output"] for p in prompts],
            "text": [p["text"] for p in prompts]
        })

    dataset_dict = DatasetDict({
        "train": create_dataset_dict(train_dataset),
        "eval": create_dataset_dict(eval_dataset)
    })

    if args.verbose:
        print("Example of a train prompt:", dataset_dict['train'][0])
        for index in random.sample(range(len(dataset_dict['train'])), 3):
            print(f"Sample {index} of train set: {dataset_dict['train'][index]}")

    return dataset_dict