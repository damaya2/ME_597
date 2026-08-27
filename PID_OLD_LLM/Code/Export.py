import ast
import json
import re


def export_raw_log_data(
    log_filepaths=[
        "logs/sft_v3_21280902.out",
    ],
    output_file="training_logs_v3.json",
):
    log_data = []
    seen_entries = set()

    # Matches any Python-style dictionary starting with standard HF/TRL metric keys
    dict_pattern = re.compile(
        r"\{'(?:loss|eval_loss|train_loss|learning_rate|epoch|grad_norm)':[^{}]+\}"
    )

    for filepath in log_filepaths:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    matches = dict_pattern.findall(line)
                    for match in matches:
                        try:
                            data_dict = ast.literal_eval(match)

                            # Derive a unique signature key (Step -> Epoch -> Num Tokens)
                            step = data_dict.get("step")
                            epoch = data_dict.get("epoch")
                            num_tokens = data_dict.get("num_tokens")
                            is_eval = "eval_loss" in data_dict

                            # Deduplication key fallback chain
                            if step is not None:
                                entry_key = (step, is_eval)
                            elif epoch is not None:
                                entry_key = (round(epoch, 4), is_eval)
                            elif num_tokens is not None:
                                entry_key = (num_tokens, is_eval)
                            else:
                                entry_key = None

                            if entry_key is not None and entry_key in seen_entries:
                                continue

                            if entry_key is not None:
                                seen_entries.add(entry_key)

                            log_data.append(data_dict)
                        except (ValueError, SyntaxError):
                            continue
        except FileNotFoundError:
            print(f"Warning: File '{filepath}' not found. Skipping...")

    # Chronological sort fallback: step -> epoch -> num_tokens
    def sort_key(x):
        return (
            x.get("step") if x.get("step") is not None else -1,
            x.get("epoch") if x.get("epoch") is not None else -1,
            x.get("num_tokens") if x.get("num_tokens") is not None else -1,
        )

    log_data.sort(key=sort_key)

    # Write the structured JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=4)

    print(f"Successfully exported {len(log_data)} log entries to {output_file}")


if __name__ == "__main__":
    export_raw_log_data()