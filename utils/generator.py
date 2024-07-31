import os
import random
from datasets import load_dataset, Dataset, DatasetDict
from LlaMAft.utils.prompter import Prompter
from LlaMAft.loader.data_module import DataCollatorForCausalLM

glue_task_to_keys = {  # Done
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

def tokenize(args, tokenizer, prompt, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=args.max_new_tokens,
        padding=False,
        return_tensors=None,
    )
    if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < args.max_new_tokens
            and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
    result["labels"] = result["input_ids"].copy()
    return result

def generate_and_tokenize_prompt(args, tokenizer, prompter, data_point):
    full_prompt = prompter.generate_prompt(
        data_point["instruction"],
        data_point["input"],
        data_point["output"],
    )
    tokenized_full_prompt = tokenize(args, tokenizer, full_prompt)
    if not args.train_on_source:
        user_prompt = prompter.generate_prompt(
            data_point["instruction"], data_point["input"]
        )
        tokenized_user_prompt = tokenize(
            args, tokenizer, user_prompt
        )
        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        if True: #add_eos_token:
            user_prompt_len -= 1
        tokenized_full_prompt["labels"] = [
                                              -100
                                          ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                user_prompt_len:
                                                                ]  # could be sped up, probably
    return tokenized_full_prompt


def data_handler(args, model, tokenizer):

    task_to_keys = glue_task_to_keys
    prompter = Prompter(args.prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    
    if args.task_name is not None:
        raw_datasets = load_dataset(
            "glue", args.task_name, cache_dir=args.cache_dir
        )
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names  # type: ignore
            num_labels = len(label_list)
    else:
        raw_datasets = load_dataset(
            "glue", "cola", cache_dir=args.cache_dir
        )
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]  # type: ignore
        if not is_regression:
            label_list = raw_datasets["train"].unique("label")  # type: ignore
            label_list.sort()
            num_labels = len(label_list)

    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        sentence1_key, sentence2_key = "sentence1", "sentence2"

    train_dataset = raw_datasets["train"] # type: ignore
    eval_dataset = raw_datasets["validation_matched" if args.task_name == "mnli" else "validation"] # type: ignore

    def create_instructions(data, sentence1_key, sentence2_key, task_labels, prompter):
        instructions = []
        for x, y, z in zip(data[sentence1_key], data[sentence2_key], data['label']):
            instruction = "Classify the relationship between the following sentences:"
            input_text = f"Premise: {x}\nHypothesis: {y}"
            output_text = task_labels[z]
            prompt = prompter.generate_prompt(instruction, input_text, output_text)
            instructions.append(prompt)
        return instructions

    train_instructions = create_instructions(train_dataset, sentence1_key, sentence2_key, id_to_label[args.task_name], prompter)
    validation_instructions = create_instructions(eval_dataset, sentence1_key, sentence2_key, id_to_label[args.task_name], prompter)

    ds_train = Dataset.from_dict({"text": train_instructions})
    ds_validation = Dataset.from_dict({"text": validation_instructions})
    instructions_ds_dict = DatasetDict({"train": ds_train, "eval": ds_validation})


    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=args.predict_with_generate,
    )

    if args.verbose:
        print("Instructions Train set:")
        print(instructions_ds_dict["train"]['text'][0])
        for index in random.sample(range(len(train_dataset)), 3):
            print(f"Sample {index} of train set: {train_dataset[index]}")

    return dict(
        train_dataset=ds_train if args.do_train else None,
        eval_dataset=ds_validation if args.do_eval else None,
        predict_dataset=ds_validation if args.do_predict else None,
        data_collator=data_collator
    )