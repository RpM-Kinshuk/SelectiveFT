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