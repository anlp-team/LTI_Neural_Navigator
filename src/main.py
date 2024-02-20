import argparse

from rag.doc_reader import QuestionAnsweringModel, ContextSelector, AnswerProcessor
from rag.embedder import EmbeddingModel, Embedder
from rag.retriever import DocumentDatabase, Ranker, Retriever


def main():
    # Initialize components
    embedding_model = EmbeddingModel()
    embedder = Embedder(embedding_model)
    database = DocumentDatabase()
    ranker = Ranker()
    retriever = Retriever(database, ranker)
    qa_model = QuestionAnsweringModel()

    # Process a query
    query = "What is the capital of France?"
    query_embedding = embedder.get_embedding(query)
    documents = retriever.retrieve(query_embedding)
    context = ContextSelector().select(documents, query)
    answer = qa_model.answer(context, query)
    processed_answer = AnswerProcessor().process(answer)

    print(processed_answer)


def arg_parser():
    parser = argparse.ArgumentParser(description="LTI Neural Navigator")
    parser.add_argument("--doc_path", type=str, default="../data/sample", help="Path to knowledge base")
    parser.add_argument("--model_ckpt", type=str, default="../model_ckpts/????", help="Path to model checkpoint")
    return parser.parse_args()


if __name__ == "__main__":
    main()
