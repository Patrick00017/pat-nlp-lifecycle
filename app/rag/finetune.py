from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GenerationConfig,
    TrainingArguments,
    Trainer,
)
import torch
import time
import evaluate
import pandas as pd
import numpy as np
from peft import LoraConfig, get_peft_model, TaskType

model_name = "google/flan-t5-base"

original_model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_function(example):
    start_prompt = "Summarize the following conversation.\n\n"
    end_prompt = "\n\nSummary: "
    prompt = [start_prompt + dialogue + end_prompt for dialogue in example["dialogue"]]
    example["input_ids"] = tokenizer(
        prompt, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids
    example["labels"] = tokenizer(
        example["summary"], padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids

    return example


dataset = load_dataset("nyamuda/samsum")
# The dataset contains 3 different splits. Tokenize function is handling all of these splits
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(
    [
        "id",
        "dialogue",
        "summary",
    ]
)
tokenized_datasets = tokenized_datasets.filter(
    lambda example, index: index % 12 == 0, with_indices=True
)

# Setting up the configuration
lora_config = LoraConfig(
    r=32,  # Rank of the low-rank matrices
    lora_alpha=32,  # Similar to learning rate
    target_modules=["q", "v"],  # Targeting query and key layers
    lora_dropout=0.05,  # Similar to dropout in neural networks
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,  # FLAN-T5 task type
)

peft_model = get_peft_model(original_model, lora_config)

output_dir = f"./peft-training-{str(int(time.time()))}"

peft_training_args = TrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,  # Automatically computes the largest batch size possible
    learning_rate=1e-3,  # Will be higher compared to LR for full finetuning
    weight_decay=0.01,
    num_train_epochs=10,
    logging_steps=50,
)

peft_trainer = Trainer(
    model=peft_model,
    args=peft_training_args,
    train_dataset=tokenized_datasets["train"],
)

time1 = time.time()
peft_trainer.train()  # Starts the training
time2 = time.time()

training_time = time2 - time1

print(
    f"Time taken to train the model for 10 epochs using LoRA is: {training_time} seconds"
)

index = 75
dialogue = dataset["test"][index]["dialogue"]
baseline_human_summary = dataset["test"][index]["summary"]

prompt = f"""
Summarize the following conversation.
 
{dialogue}
 
Summary: """

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

original_model_outputs = original_model.generate(
    input_ids=input_ids,
    generation_config=GenerationConfig(max_new_tokens=200, num_beams=1),
)
original_model_text_output = tokenizer.decode(
    original_model_outputs[0], skip_special_tokens=True
)

peft_model_outputs = peft_model.generate(
    input_ids=input_ids,
    generation_config=GenerationConfig(max_new_tokens=200, num_beams=1),
)
peft_model_text_output = tokenizer.decode(
    peft_model_outputs[0], skip_special_tokens=True
)

dash_line = "\n---------------------------------------------------------\n"
print(f"PROMPT: \n {prompt}")
print(dash_line)
print(f"BASELINE HUMAN SUMMARY:\n{baseline_human_summary}")
print(dash_line)
print(f"ORIGINAL MODEL:\n{original_model_text_output}")
print(dash_line)
print(f"PEFT MODEL:\n {peft_model_text_output}")
