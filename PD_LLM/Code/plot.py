import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def exp_decay(x, a, b, c):
    """Exponential decay model function: y = a * exp(-b * x) + c."""
    return a * np.exp(-b * x) + c


def power_law(x, a, b, c):
    """Power law decay model function: y = a * (x ** -b) + c."""
    x_safe = np.maximum(x, 1.0)
    return a * (x_safe**-b) + c


def fit_best_decay_curve(x_data, y_data):
    """Fits Power Law and Exponential Decay with realistic baseline loss bounds."""
    skip_idx = max(1, int(len(x_data) * 0.05))

    x_fit = np.asarray(x_data[skip_idx:], dtype=float)
    y_fit = np.asarray(y_data[skip_idx:], dtype=float)

    if len(x_fit) < 4:
        return None

    results = {}
    min_y, max_y = np.min(y_fit), np.max(y_fit)
    y_range = max_y - min_y

    # Force asymptote 'c' to stay within reasonable bounds of observed minimum
    # (Loss cannot asymptote below ~50% of the minimum observed evaluation loss)
    c_min_bound = min_y * 0.50
    c_max_bound = min_y * 0.98

    # --- Model 1: Constrained Power Law Decay ---
    try:
        p0_pow = [y_range * 2.0, 0.5, min_y * 0.8]
        bounds_pow = (
            [1e-4, 0.01, c_min_bound],  # Enforce lower bound on c!
            [np.inf, 2.0, c_max_bound],
        )

        popt_pow, _ = curve_fit(
            power_law, x_fit, y_fit, p0=p0_pow, bounds=bounds_pow, maxfev=20000
        )
        a_pow, b_pow, c_pow = popt_pow

        y_pred_pow = power_law(x_fit, a_pow, b_pow, c_pow)
        r2_pow = 1.0 - (
            np.sum((y_fit - y_pred_pow) ** 2)
            / (np.sum((y_fit - np.mean(y_fit)) ** 2) + 1e-8)
        )

        results["power_law"] = {
            "type": "Power Law Decay",
            "c": c_pow,
            "r2": r2_pow,
            "eval_func": lambda x: power_law(x, a_pow, b_pow, c_pow),
            "label": f"Power Law (R²={r2_pow:.3f}): {a_pow:.2f}x^(-{b_pow:.2f}) + {c_pow:.4f}",
        }
    except Exception:
        pass

    # --- Model 2: Constrained Exponential Decay ---
    try:
        p0_exp = [y_range, 0.005, min_y * 0.8]
        bounds_exp = ([1e-4, 1e-5, c_min_bound], [np.inf, 0.1, c_max_bound])

        popt_exp, _ = curve_fit(
            exp_decay, x_fit, y_fit, p0=p0_exp, bounds=bounds_exp, maxfev=20000
        )
        a_exp, b_exp, c_exp = popt_exp

        y_pred_exp = exp_decay(x_fit, a_exp, b_exp, c_exp)
        r2_exp = 1.0 - (
            np.sum((y_fit - y_pred_exp) ** 2)
            / (np.sum((y_fit - np.mean(y_fit)) ** 2) + 1e-8)
        )

        results["exponential"] = {
            "type": "Exponential Decay",
            "c": c_exp,
            "r2": r2_exp,
            "eval_func": lambda x: exp_decay(x, a_exp, b_exp, c_exp),
            "label": f"Exp Fit (R²={r2_exp:.3f}): {a_exp:.2f}e^(-{b_exp:.4f}x) + {c_exp:.4f}",
        }
    except Exception:
        pass

    if not results:
        return None

    return max(results.values(), key=lambda item: item["r2"])

def plot_metrics_separate_windows():
    # 1. Load log datasets
    with open("training_logs3.json", "r") as f:
        raw_logs = json.load(f)

    train_data = []
    eval_data = []
    step_counter = 0

    # 2. Extract metrics indexed by continuous step sequence
    for entry in raw_logs:
        if "loss" in entry:
            step_counter = entry.get("step", step_counter + 1)
            entry["step"] = step_counter
            train_data.append(entry)
        elif "eval_loss" in entry:
            entry["step"] = entry.get("step", step_counter)
            eval_data.append(entry)

    train_df = pd.DataFrame(train_data)
    eval_df = pd.DataFrame(eval_data)

    # 3. Metric configurations
    metrics_config = [
        # --- Training Metrics ---
        {
            "title": "Training Loss",
            "y_label": "Loss",
            "col": "loss",
            "df": train_df,
            "color": "#1f77b4",
            "fit_decay": True,
            "file": "metric_train_loss.png",
        },
        {
            "title": "Gradient Norm",
            "y_label": "Grad Norm",
            "col": "grad_norm",
            "df": train_df,
            "color": "#d62728",
            "file": "metric_grad_norm.png",
        },
        {
            "title": "Learning Rate",
            "y_label": "Learning Rate",
            "col": "learning_rate",
            "df": train_df,
            "color": "#ff7f0e",
            "file": "metric_learning_rate.png",
        },
        {
            "title": "Training Entropy",
            "y_label": "Entropy",
            "col": "entropy",
            "df": train_df,
            "color": "#9467bd",
            "file": "metric_train_entropy.png",
        },
        {
            "title": "Training Number of Tokens",
            "y_label": "Tokens Count",
            "col": "num_tokens",
            "df": train_df,
            "color": "#8c564b",
            "file": "metric_train_num_tokens.png",
        },
        {
            "title": "Training Mean Token Accuracy",
            "y_label": "Accuracy (%)",
            "col": "mean_token_accuracy",
            "df": train_df,
            "color": "#2ca02c",
            "scale": 100.0,
            "file": "metric_train_mean_token_accuracy.png",
        },
        # --- Evaluation Metrics ---
        {
            "title": "Eval Loss",
            "y_label": "Eval Loss",
            "col": "eval_loss",
            "df": eval_df,
            "color": "#17becf",
            "marker": "o",
            "fit_decay": True,
            "file": "metric_eval_loss.png",
        },
        {
            "title": "Eval Runtime",
            "y_label": "Runtime (s)",
            "col": "eval_runtime",
            "df": eval_df,
            "color": "#bcbd22",
            "marker": "o",
            "file": "metric_eval_runtime.png",
        },
        {
            "title": "Eval Samples Per Second",
            "y_label": "Samples / Sec",
            "col": "eval_samples_per_second",
            "df": eval_df,
            "color": "#e377c2",
            "marker": "o",
            "file": "metric_eval_samples_per_second.png",
        },
        {
            "title": "Eval Steps Per Second",
            "y_label": "Steps / Sec",
            "col": "eval_steps_per_second",
            "df": eval_df,
            "color": "#7f7f7f",
            "marker": "o",
            "file": "metric_eval_steps_per_second.png",
        },
        {
            "title": "Eval Entropy",
            "y_label": "Eval Entropy",
            "col": "eval_entropy",
            "df": eval_df,
            "color": "#9467bd",
            "marker": "o",
            "file": "metric_eval_entropy.png",
        },
        {
            "title": "Eval Number of Tokens",
            "y_label": "Eval Tokens",
            "col": "eval_num_tokens",
            "df": eval_df,
            "color": "#8c564b",
            "marker": "o",
            "file": "metric_eval_num_tokens.png",
        },
        {
            "title": "Eval Mean Token Accuracy",
            "y_label": "Accuracy (%)",
            "col": "eval_mean_token_accuracy",
            "df": eval_df,
            "color": "#2ca02c",
            "scale": 100.0,
            "file": "metric_eval_mean_token_accuracy.png",
        },
    ]

    # 4. Create and plot each distinct metric figure window
    for cfg in metrics_config:
        df, col = cfg["df"], cfg["col"]

        if df.empty or col not in df.columns:
            print(f"Skipping {cfg['title']}: column '{col}' not found.")
            continue

        valid_df = df.dropna(subset=[col, "step"])
        if valid_df.empty:
            print(f"Skipping {cfg['title']}: no valid data points.")
            continue

        scale = cfg.get("scale", 1.0)
        marker = cfg.get("marker", None)
        x_data = valid_df["step"].values
        y_data = (valid_df[col] * scale).values

        plt.figure(num=cfg["title"], figsize=(8, 5), dpi=150)

        plt.plot(
            x_data,
            y_data,
            label=f"Raw {cfg['title']}",
            color=cfg["color"],
            linewidth=1.8,
            marker=marker,
            markersize=5 if marker else None,
        )

        # 5. Best-Fit Decay Calculation
        if cfg.get("fit_decay", False) and len(x_data) >= 4:
            fit_res = fit_best_decay_curve(x_data, y_data)

            if fit_res is not None:
                x_fit = np.linspace(np.min(x_data), np.max(x_data), 350)
                y_fit = fit_res["eval_func"](x_fit)
                asymptote_val = fit_res["c"]

                # Plot curve fit line
                plt.plot(
                    x_fit,
                    y_fit,
                    color="#000000",
                    linestyle="--",
                    linewidth=2.0,
                    label=fit_res["label"],
                )

                # Plot horizontal asymptote
                plt.axhline(
                    y=asymptote_val,
                    color="#d62728",
                    linestyle=":",
                    linewidth=1.8,
                    label=f"Asymptote (y = {asymptote_val:.4f})",
                )

                plt.text(
                    x_fit[-1],
                    asymptote_val,
                    f"  y = {asymptote_val:.4f}",
                    color="#d62728",
                    verticalalignment="center",
                    fontweight="bold",
                    fontsize=9,
                )

                print("\n==========================================")
                print(
                    f" FIT RESULTS ({fit_res['type'].upper()}): {cfg['title']}"
                )
                print("==========================================")
                print(f"  Model Type               : {fit_res['type']}")
                print(f"  R^2 Score                : {fit_res['r2']:.4f}")
                print(f"  Horizontal Asymptote (y) : {asymptote_val:.6f}")
                print("==========================================\n")

        plt.xlabel("Step", fontsize=11, fontweight="bold")
        plt.ylabel(cfg["y_label"], fontsize=11, fontweight="bold")
        plt.title(
            f"{cfg['title']} vs. Step", fontsize=12, fontweight="bold"
        )
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(fontsize=9)
        plt.tight_layout()

        plt.savefig(cfg["file"], dpi=300, bbox_inches="tight")
        print(f"Saved: {cfg['file']}")

    plt.show()


if __name__ == "__main__":
    plot_metrics_separate_windows()