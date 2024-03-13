import argparse

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import json
import os
import random


def arg_parser():
    parser = argparse.ArgumentParser(description="Fine-tune the embedding model.")
    parser.add_argument("--model_id", type=str, default="mixedbread-ai/mxbai-embed-large-v1",
                        help="Model ID for the embedding model")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train the model")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--evaluation_steps", type=int, default=1, help="Number of steps to evaluate the model")
    parser.add_argument("--train_data", type=list,
                        default=["/home/ubuntu/rag-project/embedder_dataset/",
                                 "/home/ubuntu/rag-project/dataset_with_ref/"],
                        help="List of directories to train the model on")
    parser.add_argument("--sample_file", type=int, default=100,
                        help="Number of files to sample from each directory, -1 for all files")
    parser.add_argument("--sample_question", type=int, default=3,
                        help="Number of questions to sample from each file, -1 for all questions")
    parser.add_argument("--output_dir", type=str, default="/home/ubuntu/experiments/",
                        help="Directory to store the trained model and evaluation results")
    parser.add_argument("--hf_save_model_id", type=str, default="mxbai-embed-large-v1-finetuned-qa",
                        help="Save model to Hugging Face model hub with this ID")
    parser.add_argument("--cache_dir", type=str, default="/mnt/datavol/cache",
                        help="Directory to store cache")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    args = parser.parse_args()
    return args


def main_worker(args):
    model = SentenceTransformer(args.model_id, cache_folder=args.cache_dir)
    all_examples = []
    for dir_path in args.train_data:
        all_examples.extend(json_to_examples(dir_path, args.sample_file, args.sample_question))

    train_examples = all_examples[:int(len(all_examples) * 0.8)]
    test_examples = all_examples[int(len(all_examples) * 0.8):]

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    warmup_steps = int(len(train_examples) * args.epochs / args.batch_size * 0.1)
    print(f"Warmup-steps: {warmup_steps}")

    # train and output evaluation results
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_examples, name="finetune-eval")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        evaluator=evaluator,
        evaluation_steps=args.evaluation_steps,
        output_path=args.output_dir,
    )

    # Save the model to disk
    model.save(args.output_dir)

    # upload model to Hugging Face model hub
    try:
        model.save_to_hub(args.hf_save_model_id)
    except Exception as e:
        print(f"Error uploading model to Hugging Face model hub: {e}")


def json_to_examples(dir_path, sample_file=100, sample_question=3):
    examples = []
    all_files = os.listdir(dir_path)
    random.shuffle(all_files)
    if sample_file > 0:
        all_files = all_files[:sample_file]

    for filename in all_files:
        if filename.endswith(".json"):
            with open(os.path.join(dir_path, filename), "r") as f:
                data = json.load(f)
                qa_list = data["qa_list"]
                context = data["doc_text"]
                if sample_question > 0:
                    qa_list = random.sample(qa_list, min(sample_question, len(qa_list)))

                for qa_pair in qa_list:
                    if "question" not in qa_pair or "answer" not in qa_pair:
                        continue
                    question = qa_pair["question"]
                    answer = qa_pair["answer"]

                    if len(question) == 0:
                        continue
                    if "\n" in question:
                        question = question.replace("\n", "; ")
                    assert question[-1] != "\n"

                    if "ref_chunk" in qa_pair:
                        ref_chunk = qa_pair["ref_chunk"]
                        context = ref_chunk
                    if "top_k_docs" in qa_pair:
                        top_k_docs = qa_pair["top_k_docs"]
                        top_k_list = [value for key, value in top_k_docs.items()]
                        context = "\n".join(top_k_list[:3])
                        # context = top_k_list[:3]  # use only the first two contexts

                    print(f"context size: {len(context)} vs {len(data['doc_text'])}")
                    examples.append(InputExample(texts=[question, context]))

    return examples


def main():
    args = arg_parser()
    main_worker(args)


if __name__ == "__main__":
    main()
