import argparse
import os

import random
import time
import warnings
from typing import List
from tqdm import tqdm
import evaluate
import json
import os
import torch
seed1 = 20240313
random.seed(seed1)
torch.manual_seed(seed1)




def get_references(indices_path, ground_truth_path):
    with open(ground_truth_path, 'r') as f:
        ground_truth = f.readlines()

    with open(indices_path, 'r') as f:
        indices = f.readlines()
        evaluation_references = [ground_truth[int(idx.strip())] for idx in indices] 

    return evaluation_references


def calculate_bleu_rouge(input_prediction_path, evaluation_references):

    with open(input_prediction_path, mode='r') as f:
        predictions = f.readlines()
        prediction_len = len(predictions)

    evaluation_references = evaluation_references[:prediction_len]
    bleu = evaluate.load("bleu")
    rouge = evaluate.load('rouge')
    bleu_score = bleu.compute(predictions=predictions, references=evaluation_references)
    rouge_score = rouge.compute(predictions=predictions, references=evaluation_references)
    print(bleu_score['bleu'])
    print(rouge_score)

evaluation_references = get_references('data/model_evaluation/eval_indices.txt', 'data/QAparallel/test_answers.txt')
print(len(evaluation_references))

result_dir = "data/model_evaluation/"
for filename in os.listdir(result_dir):
    if filename == 'eval_indices.txt':
        continue
    file_path = os.path.join(result_dir, filename)
    print('Evaluating file: ', filename)
    calculate_bleu_rouge(file_path, evaluation_references)

