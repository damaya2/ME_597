import os
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_PATH = "crtbp_llm_dataset_20260822_031622.jsonl"
OUTPUT_DIR = "./results_llama3_1_8b_crtbp_fast_v1"
RESUME_TRAINING = False

def main():
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print("Hugging Face API token detected.")

    print(f"--- Loading CRTBP Dataset ({DATASET_PATH}) ---")
    full_dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    dataset_split = full_dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"].select(
        range(min(5000, len(dataset_split["test"])))
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True, token=hf_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def format_chat_template(example):
        text = tokenizer.apply_chat_template(
            example["messages"], 
            tokenize=False, 
            add_generation_prompt=False
        )
        return {"text": text}

    print("--- Formatting Chat Templates ---")
    train_dataset = train_dataset.map(
        format_chat_template, 
        remove_columns=train_dataset.column_names,
        num_proc=8
    )
    eval_dataset = eval_dataset.map(
        format_chat_template, 
        remove_columns=eval_dataset.column_names,
        num_proc=8
    )

    print(f"--- Loading Base Model in Native bfloat16: {MODEL_NAME} ---")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        token=hf_token,
        device_map=None,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        max_length=512,
        packing=True,
        dataset_text_field="text",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=3.0,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        args=sft_config,
        processing_class=tokenizer,
    )

    if RESUME_TRAINING:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    final_path = os.path.join(OUTPUT_DIR, "final_adapter")
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print("Training finished successfully!")

if __name__ == "__main__":
    main()