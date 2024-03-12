from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import json
import os

model_id = "mixedbread-ai/mxbai-embed-large-v1"
model = SentenceTransformer(model_id)


def json_to_examples(dir_path="./annotated_sample/"):
    examples = []
    for filename in os.listdir(dir_path):
        if filename.endswith(".json"):
            with open(os.path.join(dir_path, filename), "r") as f:
                data = json.load(f)
                qa_list = data["qa_list"]
                context = data["doc_text"]
                for pair in qa_list:
                    if "question" not in pair or "answer" not in pair:
                        print(f"Invalid pair: {pair} in {filename}")
                        continue
                    query = pair["question"]
                    answer = pair["answer"]
                    if isinstance(answer, list): # some answers are lists
                        answer = [str(a) for a in answer] # some answer items are list dicts
                        answer = "; ".join(answer)
                    examples.append(InputExample(texts=[query, context]))

    train_examples = examples[:int(len(examples) * 0.8)]
    test_examples = examples[int(len(examples) * 0.8):]
    return train_examples, test_examples


train_examples, test_examples = json_to_examples()
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