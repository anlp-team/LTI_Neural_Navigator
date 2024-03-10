from sentence_transformers import SentenceTransformer, InputExample
import json

model_id = "mixedbread-ai/mxbai-embed-large-v1"
model = SentenceTransformer(model_id)


def json_to_examples(dir_path):
    raise NotImplementedError("Implement this function")
    examples = []
    with open(dir_path, "r") as f:
        data = json.load(f)


all_pairs = json_to_examples(
    "path/to/json"
)  # List of tuples (query, context) / (query, answer)
train_examples = []

for pair in all_pairs:
    query, context = pair
    train_examples.append(InputExample(texts=[query, context]))


model.fit(train_examples)
model.save("path/to/save")


