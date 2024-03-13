# plot multi bar chart for the evaluation results

import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict

from cal_stats import extract_metric_from_file


def plot_bar_chart(
    bars, title, xlabel, ylabel, filename, xticks=("Recall", "Exact Match", "F1")
):
    x = np.arange(len(xticks))
    width = 0.2

    # set fonts
    plt.rcParams.update({"font.size": 14})
    plt.rcParams.update({"font.family": "serif"})

    fig, ax = plt.subplots()
    for i, (label, values) in enumerate(bars.items()):
        means = [v[0] for v in values]
        stds = [v[1] for v in values]
        ax.bar(x + i * width, means, width, yerr=stds, label=label)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x + width)
    ax.set_xticklabels(xticks)
    ax.legend()

    fig.tight_layout()
    plt.savefig(filename)


def main():
    metrics = ["Recall:", "F1:"]  # "Recall:", "Exact Match:", "F1:"
    xticks = ("Recall", "F1")

    root = "/home/ubuntu/experiments/testOutputs"
    dirs = [
        f"{root}/exp3_raw_emb_eval_old_prompt",
        f"{root}/exp4_w_emb_eval",
        f"{root}/exp6_chat_finetune_eval",
        f"{root}/exp7_emb_chat_finetune_eval/",
    ]

    bars = {
        "raw": list(),
        "+ emb fintune": list(),
        "+ core finetune": list(),
        "+ emb + core finetune": list(),
    }

    for i, dir in enumerate(dirs):
        for file in os.listdir(dir):
            if file.endswith(".out"):
                file_path = os.path.join(dir, file)
                metrics_dict_file = extract_metric_from_file(file_path, metrics)

                for metric, values in metrics_dict_file.items():
                    bars[list(bars.keys())[i]].append((np.mean(values), np.std(values)))

    plot_bar_chart(
        bars,
        "Evaluation Results",
        "Metrics",
        "Scores",
        "evaluation_fig.png",
        xticks,
    )


if __name__ == "__main__":
    main()
