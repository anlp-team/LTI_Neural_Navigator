from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import json
import os
import random

model_id = "mixedbread-ai/mxbai-embed-large-v1"
model = SentenceTransformer(model_id, cache_folder="/mnt/portData/cache")


def json_to_examples(dir_path, sample_file=100, sample_question=3):
    examples = []
    all_files = os.listdir(dir_path)
    random.shuffle(all_files)
    all_files = all_files[:sample_file]

    for filename in all_files:
        if filename.endswith(".json"):
            with open(os.path.join(dir_path, filename), "r") as f:
                data = json.load(f)
                qa_list = data["qa_list"]
                context = data["doc_text"]
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
                        # context = "\n".join(top_k_list)
                        context = top_k_list[0]
                        
                    print(f"context size: {len(context)} vs { len(data['doc_text']) }")
                    examples.append(InputExample(texts=[question, context]))

    return examples


all_examples = []
for dir_path in ["./embedder_dataset", "./dataset_with_ref"]:
    all_examples.extend(json_to_examples(dir_path))

train_examples = all_examples[:int(len(all_examples) * 0.8)]
test_examples = all_examples[int(len(examples) * 0.8):]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.MultipleNegativesRankingLoss(model)
num_epochs = 1
warmup_steps = int(len(train_examples) * num_epochs / 16 * 0.1)
print(f"Warmup-steps: {warmup_steps}")

# train and output evaluation results
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_examples, name="finetune-eval")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    evaluator=evaluator,
    evaluation_steps=1,
    output_path="./experiments/test",
)


# Save the model
model.save("./experiments/test")
