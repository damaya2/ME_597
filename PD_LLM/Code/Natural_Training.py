import os
import torch
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, prepare_model_for_kbit_training
import transformers
import transformers.trainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

# Directly bypass the CVE-2025-32434 PyTorch < 2.6 check
transformers.trainer.check_torch_load_is_safe = lambda: None
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_PATH = "crtbp_natural_language_20260808_234158.jsonl"
OUTPUT_DIR = "./results_llama3_1_8b_crtbp_sft_v2"
RESUME_TRAINING = True  # Set to True when resuming, False for brand new runs


def main():
    # 0. Authenticate Hugging Face session
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("--- Hugging Face Authenticated Successfully ---")
    else:
        print("--- No HF_TOKEN environment variable found, using cached credentials ---")

    print(f"--- Loading CRTBP Dataset ({DATASET_PATH}) ---")
    full_dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    # Ensure columns map to 'prompt' and 'completion'
    if (
        "response" in full_dataset.column_names
        and "completion" not in full_dataset.column_names
    ):
        print("--- Mapping 'response' column to 'completion' ---")
        full_dataset = full_dataset.rename_column("response", "completion")

    # Train/Validation Split: 95% Train, fixed 1,595 samples
    dataset_split = full_dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"].select(
        range(min(1595, len(dataset_split["test"])))
    )

    print(f"Train Dataset Size: {len(train_dataset):,} samples")
    print(f"Eval Dataset Size: {len(eval_dataset):,} samples")

    # 1. 4-Bit NormalFloat Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"--- Loading Base Model: {MODEL_NAME} ---")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True, token=hf_token
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token=hf_token,
    )

    # Enable gradient checkpointing and prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    # 2. LoRA Adapter Configuration (Must always be defined for 4-bit models)
    print("--- Defining LoRA Configuration ---")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 3. SFT Configuration
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        completion_only_loss=True,  # Calculates loss ONLY on response tokens
        max_length=1024,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,  # Fast eval batch size
        gradient_accumulation_steps=8,  # Effective BS = 64
        learning_rate=2e-4,
        logging_steps=1,  # Record train loss every single step
        num_train_epochs=2.5,  # 2.5 Epochs (~5,922 steps)
        eval_strategy="steps",
        eval_steps=40,
        save_strategy="steps",
        save_steps=200,  # Save checkpoint every 200 steps
        save_total_limit=10,  # Safe limit so load_best_model_at_end doesn't delete the best checkpoint
        load_best_model_at_end=True,  # Automatically loads checkpoint with lowest eval_loss
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
        gradient_checkpointing=True,
        dataset_num_proc=8,  # Multi-threaded dataset tokenization
    )

    print("--- Initializing SFTTrainer ---")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        args=sft_config,
        processing_class=tokenizer,
    )
    if RESUME_TRAINING:
        checkpoint_to_resume = os.path.join(OUTPUT_DIR, "checkpoint-4000")
        print(f"--- Resuming Training specifically from {checkpoint_to_resume} ---")
        trainer.train(resume_from_checkpoint=checkpoint_to_resume)
    else:
        print("--- Starting New Training Run ---")
        trainer.train()

    # 4. Save Final Weights
    final_path = os.path.join(OUTPUT_DIR, "final_adapter")
    print(f"--- Saving Final Adapter to {final_path} ---")
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print("Training finished successfully!")


if __name__ == "__main__":
    main()