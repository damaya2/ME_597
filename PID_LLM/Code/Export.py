import ast
import json
import re


def export_raw_log_data(
    log_filepaths=["logs/sft_v3_21351117.out"],
    output_file="training_logs_v3.json",
):
    log_data = []
    seen_entries = set()

    # Generic pattern to catch ANY dictionary logged on a line
    dict_pattern = re.compile(r"\{[^{}]+\}")

    for filepath in log_filepaths:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    matches = dict_pattern.findall(line)
                    for match in matches:
                        try:
                            data_dict = ast.literal_eval(match)

                            # Validate that this is a trainer log dict
                            if not isinstance(data_dict, dict):
                                continue

                            # Target standard HF metrics
                            relevant_keys = {
                                "loss",
                                "eval_loss",
                                "learning_rate",
                                "epoch",
                                "step",
                            }
                            if not any(k in data_dict for k in relevant_keys):
                                continue

                            step = data_dict.get("step")
                            epoch = data_dict.get("epoch")
                            is_eval = "eval_loss" in data_dict

                            # Uniquely identify logging events
                            if step is not None:
                                entry_key = (step, is_eval)
                            elif epoch is not None:
                                entry_key = (round(epoch, 4), is_eval)
                            else:
                                entry_key = None

                            if (
                                entry_key is not None
                                and entry_key in seen_entries
                            ):
                                continue

                            if entry_key is not None:
                                seen_entries.add(entry_key)

                            log_data.append(data_dict)
                        except (ValueError, SyntaxError):
                            continue
        except FileNotFoundError:
            print(f"Warning: File '{filepath}' not found. Skipping...")

    log_data.sort(
        key=lambda x: (
            x.get("step", -1) if x.get("step") is not None else -1,
            x.get("epoch", -1) if x.get("epoch") is not None else -1,
        )
    )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=4)

    print(
        f"Successfully exported {len(log_data)} log entries to {output_file}"
    )


if __name__ == "__main__":
    export_raw_log_data()