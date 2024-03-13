import argparse
from sentence_transformers import SentenceTransformer, util


def arg_parser():
    parser = argparse.ArgumentParser(description='Cosine Similarity')
    parser.add_argument('--model_output', type=str,
                        default="",
                        help='Output file for the model')
    parser.add_argument("--ground_truth", type=str,
                        default="",
                        help="Ground truth file")
    parser.add_argument("--embedding_model_id", type=str,
                        default="",
                        help="Embedding model id")
    parser.add_argument("--batch_size", type=int,
                        default=32,
                        help="Batch size for the embedding model")
    parser.add_argument("--cache_dir", type=str,
                        default="",
                        help="Cache directory for the embedding model")
    return parser.parse_args()


def main_worker(args):
    with open(args.model_output, "r") as f:
        model_output = f.readlines()

    with open(args.ground_truth, "r") as f:
        ground_truth = f.readlines()

    # prepare embedding mdeol
    emb_model = SentenceTransformer(
        args.embedding_model_id,
        cache_dir=args.cache_dir
    )

    assert len(model_output) == len(ground_truth), "Length of model output and ground truth should be the same"

    global_cosine_scores = []

    # iterate the batches
    for i in range(0, len(model_output), args.batch_size):
        # prepare the batch
        batch_model_output = model_output[i:i + args.batch_size]
        batch_ground_truth = ground_truth[i:i + args.batch_size]

        # get the embeddings
        model_output_embeddings = emb_model.encode(batch_model_output, convert_to_tensor=True)
        ground_truth_embeddings = emb_model.encode(batch_ground_truth, convert_to_tensor=True)

        # calculate the cosine similarity
        cosine_scores = util.cos_sim(model_output_embeddings, ground_truth_embeddings)

        global_cosine_scores.extend([cosine_scores[j][j] for j in range(i, i + len(batch_model_output))])

    print("=" * 50)
    print("Metrics: ")
    print(f"Mean Cosine Similarity: {sum(global_cosine_scores) / len(global_cosine_scores):.4f}")
    print("=" * 50)


def main():
    args = arg_parser()
    main_worker(args)


if __name__ == "__main__":
    main()
