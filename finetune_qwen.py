#!/usr/bin/env python3
"""
Fine-tune Qwen3-0.6B on open-r1/codeforces-cots dataset for instruction following.
Based on Hugging Face Skills Training guide.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    HfArgumentParser,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    model_name: str = field(
        default="Qwen/Qwen2.5-0.5B-Instruct",
        metadata={"help": "Model name or path"}
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "Use LoRA for parameter-efficient fine-tuning"}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout"}
    )


@dataclass
class DataArguments:
    """Arguments for data configuration."""
    dataset_name: str = field(
        default="open-r1/codeforces-cots",
        metadata={"help": "Dataset name on Hugging Face Hub"}
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of samples for quick testing (None = use all)"}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"}
    )


@dataclass
class MyTrainingArguments:
    """Arguments for training configuration."""
    output_dir: str = field(
        default="./qwen-codeforces-sft",
        metadata={"help": "Output directory"}
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "Number of training epochs"}
    )
    per_device_train_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size per device"}
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={"help": "Gradient accumulation steps"}
    )
    learning_rate: float = field(
        default=2e-4,
        metadata={"help": "Learning rate"}
    )
    warmup_steps: int = field(
        default=100,
        metadata={"help": "Warmup steps"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Logging steps"}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "Save checkpoint steps"}
    )
    push_to_hub: bool = field(
        default=True,
        metadata={"help": "Push model to Hugging Face Hub"}
    )
    hub_model_id: Optional[str] = field(
        default=None,
        metadata={"help": "Hub model ID (default: username/qwen-codeforces-sft)"}
    )


def format_instruction(example):
    """Format dataset examples into instruction format."""
    # The open-r1/codeforces-cots dataset has 'problem' and 'solution' fields
    if 'problem' in example and 'solution' in example:
        messages = [
            {"role": "system", "content": "You are an expert competitive programmer who solves coding problems step by step."},
            {"role": "user", "content": example['problem']},
            {"role": "assistant", "content": example['solution']}
        ]
        return {"messages": messages}
    
    # Fallback for different dataset structure
    return example


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, MyTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Set up Hub model ID
    if training_args.hub_model_id is None:
        from huggingface_hub import whoami
        username = whoami()['name']
        training_args.hub_model_id = f"{username}/qwen-codeforces-sft"
    
    print("=" * 80)
    print("🚀 Fine-tuning Configuration")
    print("=" * 80)
    print(f"Model: {model_args.model_name}")
    print(f"Dataset: {data_args.dataset_name}")
    print(f"Output: {training_args.output_dir}")
    print(f"Hub Model ID: {training_args.hub_model_id}")
    print(f"Max samples: {data_args.max_samples or 'All'}")
    print(f"LoRA: {model_args.use_lora}")
    print(f"Epochs: {training_args.num_train_epochs}")
    print(f"Batch size: {training_args.per_device_train_batch_size}")
    print(f"Learning rate: {training_args.learning_rate}")
    print("=" * 80)
    
    # Load tokenizer
    print("\n📦 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name, trust_remote_code=True)
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print(f"📦 Loading model: {model_args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    
    # Apply LoRA if enabled
    if model_args.use_lora:
        print("🔧 Applying LoRA configuration...")
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Load dataset
    print(f"\n📚 Loading dataset: {data_args.dataset_name}...")
    dataset = load_dataset(data_args.dataset_name, split="train")
    
    # Limit samples for quick testing
    if data_args.max_samples:
        dataset = dataset.select(range(min(data_args.max_samples, len(dataset))))
        print(f"   Using {len(dataset)} samples for testing")
    else:
        print(f"   Total samples: {len(dataset)}")
    
    # Format dataset
    print("🔄 Formatting dataset...")
    dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)
    
    # Training configuration
    sft_config = SFTConfig(
        output_dir=training_args.output_dir,
        num_train_epochs=training_args.num_train_epochs,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        learning_rate=training_args.learning_rate,
        warmup_steps=training_args.warmup_steps,
        logging_steps=training_args.logging_steps,
        save_steps=training_args.save_steps,
        save_total_limit=3,
        push_to_hub=training_args.push_to_hub,
        hub_model_id=training_args.hub_model_id,
        hub_strategy="every_save",
        report_to=["tensorboard"],
        max_seq_length=data_args.max_seq_length,
        packing=False,
        gradient_checkpointing=True,
        bf16=torch.cuda.is_available(),
        optim="adamw_torch",
        lr_scheduler_type="cosine",
    )
    
    # Create trainer
    print("🎓 Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    # Start training
    print("\n" + "=" * 80)
    print("🏃 Starting training...")
    print("=" * 80)
    trainer.train()
    
    # Save final model
    print("\n💾 Saving final model...")
    trainer.save_model()
    
    if training_args.push_to_hub:
        print(f"\n✅ Model pushed to Hub: https://huggingface.co/{training_args.hub_model_id}")
        print("\nTo use your model:")
        print(f"```python")
        print(f"from transformers import AutoModelForCausalLM, AutoTokenizer")
        print(f"")
        print(f'model = AutoModelForCausalLM.from_pretrained("{training_args.hub_model_id}")')
        print(f'tokenizer = AutoTokenizer.from_pretrained("{training_args.hub_model_id}")')
        print(f"```")
    
    print("\n✨ Training complete!")


if __name__ == "__main__":
    main()
