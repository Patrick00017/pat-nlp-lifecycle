# LLM_fine_turning/train_psydt_lora.py
# YIRONGCHEN/PsyDTCorpus/train_psydt_lora.py
import os
import json
import argparse
from typing import List, Dict
from inspect import signature

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, TaskType, get_peft_model
from load_qa import load_and_inspect


def tokenize_function(examples, tokenizer, max_length):
    # <|im_start|>system ... <|im_end|>
    # <|im_start|>user ... <|im_end|>
    # <|im_start|>assistant ... <|im_end|>
    samples = []
    for example in examples:
        system_prompt = "<|im_start|>system ... <|im_end|>\n"
        user_prompt = f"<|im_start|>user {example['question']} <|im_end|>\n<|im_start|>assistant\n"
        label_prompt = f"{example['answer']} <|im_end|>"

        instruction_part = tokenizer(
            system_prompt + user_prompt, add_special_tokens=False
        )
        response_part = tokenizer(label_prompt, add_special_tokens=False)

        input_ids = instruction_part["input_ids"] + response_part["input_ids"]
        attention_mask = (
            instruction_part["attention_mask"] + response_part["attention_mask"]
        )
        labels = [-100] * len(instruction_part["input_ids"]) + response_part[
            "input_ids"
        ]

        # 长度截断（左截断，尽量保留结尾与完整回答）
        if len(input_ids) > max_length:
            input_ids = input_ids[-max_length:]
            attention_mask = attention_mask[-max_length:]
            labels = labels[-max_length:]

        samples.append(
            {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        )
    return samples


def load_bts_qa_dataset(path, tokenizer, max_length):
    qa = load_and_inspect(path)
    samples = tokenize_function(qa, tokenizer=tokenizer, max_length=max_length)
    return samples


def main():
    parser = argparse.ArgumentParser()
    # 模型与数据集参数
    parser.add_argument("--model_repo", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--model_local_dir", default="./Qwen/Qwen3-1.7B")

    # 训练/评估文件名（若不存在将自动下载）
    parser.add_argument(
        "--train_file", default="./PsyDTCorpus_train_mulit_turn_packing.json"
    )
    parser.add_argument(
        "--eval_file", default="./PsyDTCorpus_test_single_turn_split.json"
    )

    # 自动下载数据集所需参数
    parser.add_argument("--dataset_repo", default="YIRONGCHEN/PsyDTCorpus")
    parser.add_argument("--dataset_dir", default="./data/PsyDTCorpus")

    # 训练超参
    parser.add_argument("--output_dir", default="./output/qwen3-1_7b-psydt-lora")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--save_steps", type=int, default=400)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=200)

    # 开启评估，并限制数据量：训练集300条对话，测试集20条对话
    parser.add_argument("--do_eval", action="store_true", default=True)
    parser.add_argument(
        "--max_train_items",
        type=int,
        default=300,
        help="限制读取训练对话条数（默认300）",
    )
    parser.add_argument(
        "--max_eval_items", type=int, default=20, help="限制读取评估对话条数（默认20）"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_repo, use_fast=False, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_repo,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    model.enable_input_require_grads()

    samples = load_bts_qa_dataset("qa.json", tokenizer, max_length=1024)
    train_dataset = Dataset.from_list(samples)
    # print(samples)
    # LoRA 配置（r=8）
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)

    kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        gradient_checkpointing=True,
    )
    # 新增：每个 epoch 结束保存一次 checkpoint
    kwargs["save_strategy"] = "epoch"

    training_args = TrainingArguments(**kwargs)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print(">>> 开始训练...")
    trainer.train()


if __name__ == "__main__":
    main()
