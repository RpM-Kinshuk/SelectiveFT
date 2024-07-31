# Prompt templates

This directory contains Alpaca style prompt templates for LLAMA family of models.

## Format

A template has the following keys:

- `prompt_input`: Used when input is not None. Uses `{instruction}` and `{input}` placeholders.
- `prompt_no_input`: Used when input is None. Only uses `{instruction}` placeholders.
- `description`: Description of the template.
- `response_split`: Separator text used when cutting real response from the model output.

## Example template

The default template is `alpaca.json`

```json
{
    "description": "Default template used by Alpaca.",
    "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: ",
    "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response: ",
    "response_split": "### Response:"    
}

```

## Current templates

### alpaca

Default template used by the original alpaca repo, with no `\n` after the response field.

### alpaca_short

A shorter alpaca template which may perform just as well and spare some tokens. Models created with the default template are supposed to be queryable by the short tempalte as well.

### meta

A template used by Meta training. It is similar to the default alpaca template.
