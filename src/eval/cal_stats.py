
import os
import numpy as np
from collections import defaultdict

def extract_metric_from_file(file_path, metrics):
    metrics_dict = defaultdict(list)

    with open(file_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        for metric in metrics:
            if metric in line:
                items = line.split()

                if len(items) == len(metric.split()) + 1:
                    metric_value = float(items[-1])
                    metrics_dict[metric].append(metric_value)
    return metrics_dict

def print_agg_stats(metrics):
    for metric, values in metrics.items():
        print(f'{metric}: {np.mean(values):.4f} ± {np.std(values):.4f}')

def main():
    metrics = ['Recall:', 'Exact Match:', 'F1:']

    root = "/home/ubuntu/experiments/testOutputs"
    dirs = [f'{root}/exp3_raw_emb_eval_old_prompt', f'{root}/exp4_w_emb_eval', f'{root}/exp6_chat_finetune_eval', f'{root}/exp7_emb_chat_finetune_eval/']

    for dir in dirs:
        for file in os.listdir(dir):
            if file.endswith('.out'):
                file_path = os.path.join(dir, file)
                metrics_dict_file = extract_metric_from_file(file_path, metrics)

                print(f"\nFile: {file}")
                print_agg_stats(metrics_dict_file)

if __name__ == "__main__":
    main()