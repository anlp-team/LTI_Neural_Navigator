"""
This file is used to finetune the model on the question answering task.
The fine-tuning is done using the Huggingface Trainer API.
The fine-tuning is accelerated using the deepspeed library.
"""

import os
import json
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from accelerate import DeepSpeedPlugin, Accelerator
from langchain import hub
from datasets import load_dataset, Dataset, DatasetDict


accelerator = Accelerator()

modelpath = "meta-llama/Llama-2-7b-hf"

# Load 4-bit quantized model
model = AutoModelForCausalLM.from_pretrained(
    modelpath,
    device_map="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    ),
    # quantization_config=BitsAndBytesConfig(
    #     load_in_8bit=True,
    #     compute_dtype=torch.bfloat16,
    #     quant_type="nf4",
    # ),
    torch_dtype=torch.bfloat16,
)

# Load (slow) Tokenizer, fast tokenizer sometimes ignores added tokens
tokenizer = AutoTokenizer.from_pretrained(modelpath, use_fast=False)

# Add tokens <|im_start|> and <|im_end|>, latter is special eos token
tokenizer.pad_token = "</s>"
tokenizer.add_tokens(["<|im_start|>"])
tokenizer.add_special_tokens(dict(eos_token="<|im_end|>"))
model.resize_token_embeddings(len(tokenizer))
model.config.eos_token_id = tokenizer.eos_token_id

# Prepare model for kbit training
model = prepare_model_for_kbit_training(model)
config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "down_proj",
        "v_proj",
        "gate_proj",
        "o_proj",
        "up_proj",
    ],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["lm_head", "embed_tokens"],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
model.config.use_cache = False


def prep_data(dir_path):
    langchain_prompt = hub.pull("rlm/rag-prompt")

    data = []
    for file in os.listdir(dir_path):
        if not file.endswith(".json"):
            continue
        with open(os.path.join(dir_path, file), "r") as f:
            # json
            json_data = json.load(f)
            if "qa_list" not in json_data or "doc_text" not in json_data:
                continue
            qa_list = json_data["qa_list"]
            context = json_data["doc_text"]

            for pair in qa_list:
                if "question" not in pair or "answer" not in pair:
                    print(f"Invalid pair: {pair} in {file}")
                    continue
                query = pair["question"]
                answer = pair["answer"]
                if isinstance(answer, list):  # some answers are lists
                    answer = [
                        str(a) for a in answer
                    ]  # some answer items are list dicts
                    answer = "; ".join(answer)

                data.append(
                    langchain_prompt.format_messages(
                        question=query,
                        answer=answer,
                        context=context,
                    )[0].content
                )

    return Dataset.from_dict({"text": data})


dataset = prep_data("./annotated_sample/")
dataset = dataset.train_test_split(test_size=0.1)


# Tokenize dataset
def tokenize(element):
    return tokenizer(
        element["text"],
        truncation=True,
        max_length=512,
        add_special_tokens=False,
    )


dataset_tokenized = dataset.map(
    tokenize,
    batched=True,
    num_proc=os.cpu_count(),  # multithreaded
    remove_columns=["text"],  # don't need this anymore, we have tokens from here on
)


# collate function - to transform list of dictionaries [ {input_ids: [123, ..]}, {.. ] to single batch dictionary { input_ids: [..], labels: [..], attention_mask: [..] }
def collate(elements):
    tokenlist = [e["input_ids"] for e in elements]
    tokens_maxlen = max([len(t) for t in tokenlist])

    input_ids, labels, attention_masks = [], [], []
    for tokens in tokenlist:
        pad_len = tokens_maxlen - len(tokens)

        # pad input_ids with pad_token, labels with ignore_index (-100) and set attention_mask 1 where content otherwise 0
        input_ids.append(tokens + [tokenizer.pad_token_id] * pad_len)
        labels.append(tokens + [-100] * pad_len)
        attention_masks.append([1] * len(tokens) + [0] * pad_len)

    batch = {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention_masks),
    }
    return batch


bs = 1  # batch size
ga_steps = 1  # gradient acc. steps
epochs = 5
steps_per_epoch = len(dataset_tokenized["train"]) // (bs * ga_steps)

args = TrainingArguments(
    output_dir="out",
    per_device_train_batch_size=bs,
    per_device_eval_batch_size=bs,
    evaluation_strategy="steps",
    logging_steps=1,
    eval_steps=steps_per_epoch,
    save_steps=steps_per_epoch,
    gradient_accumulation_steps=ga_steps,
    num_train_epochs=epochs,
    lr_scheduler_type="constant",
    optim="paged_adamw_32bit",
    learning_rate=0.0002,
    group_by_length=True,
    fp16=True,
    ddp_find_unused_parameters=False,
    # deepspeed config
    # deepspeed="./src/core_models/deepspeed_config.json",
    # report_to="none",
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=collate,
    train_dataset=dataset_tokenized["train"],
    eval_dataset=dataset_tokenized["test"],
)

model.config.use_cache = False
trainer.train()
