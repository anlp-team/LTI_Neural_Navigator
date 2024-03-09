import time
from pynvml import *
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from datasets import load_dataset
from trl import SFTTrainer
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import TrainingArguments, Trainer


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


    
base_model_id = "microsoft/phi-2"
#Load the tokenizer

tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_eos_token=True, use_fast=True)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
          base_model_id, trust_remote_code=True, quantization_config=bnb_config, revision="refs/pr/23", device_map={"": 0}, torch_dtype="auto", flash_attn=True, flash_rotary=True, fused_dense=True
)
# print(print_gpu_utilization())
model = prepare_model_for_kbit_training(model) #This line yield an error if you don't use revision="refs/pr/23"
dataset = load_dataset("Babelscape/rebel-dataset")

peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules= ["Wqkv", "out_proj"]
)

training_arguments = TrainingArguments(
        output_dir="./phi2-results2",
        evaluation_strategy="steps",
        do_eval=True,
        per_device_train_batch_size=6,
        # gradient_accumulation_steps=8,
        per_device_eval_batch_size=1,
        log_level="debug",
        save_steps=10000,
        logging_steps=100, 
        learning_rate=5e-5,
        eval_steps=5000,
        optim='paged_adamw_8bit',
        # bf16 = True,
        fp16=True, #change to bf16 if are using an Ampere GPU
        num_train_epochs=0.1,
        warmup_steps=5000,
        lr_scheduler_type="linear",
)
trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        peft_config=peft_config,
        dataset_text_field="context",
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=True
)
trainer.train()