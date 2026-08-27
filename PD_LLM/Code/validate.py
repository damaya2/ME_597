import json
from transformers import AutoTokenizer

DATASET_PATH = "crtbp_natural_language_20260808_234158.jsonl"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

def validate_jsonl():
    print(f"=== Starting Audit for {DATASET_PATH} ===\n")
    
    total_lines = 0
    corrupted_lines = 0
    missing_fields = 0
    empty_content = 0
    
    keys_found = set()
    prompt_lengths = []
    completion_lengths = []

    # 1. Check Line Formatting & Keys
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            total_lines += 1
            try:
                data = json.loads(line)
                keys_found.update(data.keys())
                
                # Check for prompt/completion or prompt/response
                p_text = data.get("prompt", "")
                c_text = data.get("completion", "") or data.get("response", "")
                
                if not p_text or not c_text:
                    empty_content += 1
                
                prompt_lengths.append(len(p_text))
                completion_lengths.append(len(c_text))
                
            except json.JSONDecodeError:
                corrupted_lines += 1

    print(f"📊 Dataset Structure Overview:")
    print(f"  • Total Rows: {total_lines:,}")
    print(f"  • Corrupted JSON Lines: {corrupted_lines}")
    print(f"  • Fields Found Across Dataset: {list(keys_found)}")
    print(f"  • Rows with Missing/Empty Content: {empty_content}")

    if corrupted_lines > 0:
        print("\n❌ CRITICAL: Fix corrupted JSON lines before training!")
        return

    # 2. Verify Schema for SFTTrainer
    print("\n🔍 Field Mapping Check:")
    has_prompt = "prompt" in keys_found
    has_completion = "completion" in keys_found or "response" in keys_found
    
    if has_prompt and has_completion:
        print("  ✅ Columns match SFTTrainer requirements for `completion_only_loss=True`.")
    else:
        print("  ❌ WARNING: Dataset lacks 'prompt' and 'completion'/'response' keys.")

    # 3. Token Length Audit (Checking max_length=1024 fit)
    print("\n📏 Tokenization & Truncation Check (using Llama-3.1 Tokenizer)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    oversized_count = 0
    sample_check_limit = min(total_lines, 2000)  # Check sample set for speed
    
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for i in range(sample_check_limit):
            data = json.loads(f.readline())
            p = data.get("prompt", "")
            c = data.get("completion", "") or data.get("response", "")
            
            # Estimate combined token length
            total_tokens = len(tokenizer.encode(p + c))
            if total_tokens > 1024:
                oversized_count += 1

    pct_oversized = (oversized_count / sample_check_limit) * 100
    print(f"  • Estimated tokens checked across first {sample_check_limit:,} rows.")
    print(f"  • Rows exceeding max_length=1024: {oversized_count} ({pct_oversized:.1f}%)")
    if pct_oversized > 5:
        print("    ⚠️ Notice: Some completions will be truncated during training.")

    # 4. Print Sample Rows
    print("\n Visual Inspection (Sample Row 1):")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        sample = json.loads(f.readline())
        print(f"PROMPT:\n{sample.get('prompt', '')[:300]}...")
        print(f"\nCOMPLETION:\n{sample.get('completion', sample.get('response', ''))[:300]}...")

if __name__ == "__main__":
    validate_jsonl()