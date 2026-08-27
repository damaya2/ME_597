import ast
import json
import re


def export_raw_log_data(
    log_filepaths=[
        "logs/sft_llama_21050232.out",  # Run 1 (Steps 0 -> 4000)
        "logs/sft_llama_21092190.out",  # Run 2 (Steps 4000 -> 5923, Completed)
    ],
    output_file="training_logs4.json",
):
    log_data = []
    seen_steps = set()

    # Match ANY metric dictionary printed in Hugging Face / TRL logs
    dict_pattern = re.compile(
        r"\{'(?:loss|eval_loss|learning_rate|epoch)'.*?\}"
    )

    for filepath in log_filepaths:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    matches = dict_pattern.findall(line)
                    for match in matches:
                        try:
                            data_dict = ast.literal_eval(match)
                            
                            # Filter out exact duplicate steps if checkpoint overlaps
                            step = data_dict.get("step")
                            is_eval = "eval_loss" in data_dict
                            entry_key = (step, is_eval)

                            if step is not None and entry_key in seen_steps:
                                continue

                            if step is not None:
                                seen_steps.add(entry_key)

                            log_data.append(data_dict)
                        except (ValueError, SyntaxError):
                            continue
        except FileNotFoundError:
            print(f"Warning: File '{filepath}' not found. Skipping...")

    # Sort sequentially by step
    log_data.sort(key=lambda x: x.get("step", 0))

    # Write the structured JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=4)

    print(f"Successfully exported {len(log_data)} log entries to {output_file}")


if __name__ == "__main__":
    export_raw_log_data()