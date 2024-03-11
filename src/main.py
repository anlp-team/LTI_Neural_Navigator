import argparse
import os
import warnings
from typing import List

import torch
from langchain import hub
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, pipeline

from rag.doc_reader import QuestionAnsweringModel, ContextSelector, AnswerProcessor
from rag.embedder import EmbeddingModel, Embedder
from rag.retriever import DocumentDatabase, Ranker, Retriever


def arg_parser():
    parser = argparse.ArgumentParser(description="LTI Neural Navigator")
    # parser.add_argument("--question", type=str,
    #                     default="Where is CMU located?",
    #                     help="Question to ask")
    parser.add_argument("--test_set", type=str,
                        default="/home/ubuntu/rag-project/testset/final_test.txt",
                        help="Path to test set")
    parser.add_argument("--test_output", type=str,
                        default="/home/ubuntu/rag-project/testset/final_test_output.txt",
                        help="Path to test output")
    parser.add_argument("--topk", type=int, default=5, help="Maximum number of documents to retrieve")
    parser.add_argument("--generate_batch_size", type=int, default=10,
                        help="Batch size for generation, 0 for no batching")
    parser.add_argument("--max_tokens", type=int, default=150,
                        help="Maximum new tokens to generate")
    parser.add_argument("--search_type", type=str, default="mmr", help="Type of search to perform",
                        choices=["similarity", "mmr", "similarity_score_threshold"])
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")

    parser.add_argument("--knowledge_base", type=str, default="/home/ubuntu/rag-project/data/sample",
                        help="Path to knowledge base")
    parser.add_argument("--core_model_id", type=str,
                        default="meta-llama/Llama-2-7b-chat-hf",
                        # default="microsoft/phi-2",
                        help="Model ID for the core model")
    parser.add_argument("--embed_model_id", type=str,
                        default="mixedbread-ai/mxbai-embed-large-v1",
                        # default="BAAI/bge-small-en",
                        help="Model ID for the embedding model")

    parser.add_argument("--cache_dir", type=str, default="/mnt/datavol/cache", help="Directory to store cache")
    parser.add_argument("--streaming_output", action="store_true", default=False, help="Stream output to console")

    return parser.parse_args()


def main():
    # NOT IMPLEMENTED
    raise NotImplementedError

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


# Temporarily solution
def langchain(args):
    loader = DirectoryLoader(args.knowledge_base)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)

    # prepare embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name=args.embed_model_id,
        model_kwargs={
            "device": args.device,
        },
        encode_kwargs={

        },
        cache_folder=args.cache_dir
    )

    # prepare vector store
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # prepare core model
    tokenizer = AutoTokenizer.from_pretrained(
        args.core_model_id,
        use_fast=True,
        cache_dir=args.cache_dir
    )
    core_model = AutoModelForCausalLM.from_pretrained(
        args.core_model_id,
        torch_dtype=torch.bfloat16,
        load_in_8bit=True,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto",
        cache_dir=args.cache_dir
    )
    pipe = pipeline(
        task="text-generation",
        model=core_model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_tokens,
        device=None  # because we use device_map="auto"
    )
    gen_pipeline = HuggingFacePipeline(pipeline=pipe)

    # prepare prompt
    prompt = hub.pull("yeyuan/rag-prompt-llama")

    if args.generate_batch_size > 0:
        # Batch generation mode
        # Define the chain
        # TODO: add reranker to chain
        chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | gen_pipeline
                | StrOutputParser()
        )

        # clear old output
        with open(args.test_output, "w") as f:
            pass

        # get questions
        with open(args.test_set, "r") as f:
            questions = f.readlines()
            questions = [q.strip() for q in questions]

            # get iterator of questions
            for batch_questions in get_data(questions, args.generate_batch_size):
                # print questions
                print("\n")
                print("=" * 50)
                print(f"Questions in the batch: {len(batch_questions)}")
                for i, q in enumerate(batch_questions):
                    print(f"[{i}] {q}")

                # Run the chain
                outputs = chain.batch(batch_questions)

                # sanitize outputs
                outputs = [o.replace('\n', '').strip() for o in outputs]

                with open(args.test_output, "a") as fout:
                    for output in outputs:
                        fout.write(output + "\n")

                # print answers
                for i, output in enumerate(outputs):
                    print(f"[{i}] {batch_questions[i]}")
                    print(f"Answer: {output}")

    else:
        # interactive
        while True:
            question = input("What's your problem today (input \"exit\" to exit): ")
            if question == "exit":
                break

            # Define the chain
            # TODO: add reranker to chain
            chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | gen_pipeline
                    | StrOutputParser()
            )

            if args.streaming_output:
                # Stream the chain
                # print("WARNING: Streaming output might not work well with some models.")
                print("Answer:")
                for ch in chain.stream(question):
                    print(ch, end='')

            else:
                # Run the chain
                output = chain.invoke(question)

                print("Answer:", output)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def format_docs_and_print(docs):
    s = "\n\n".join(doc.page_content for doc in docs)
    print(s)
    return s


def get_data(data: List[str], batch_size: int):
    questions_iter = iter(data)
    while True:
        batch_questions = []
        for _ in range(batch_size):
            try:
                question = next(questions_iter)
                batch_questions.append(question)
            except StopIteration:
                break
        if not batch_questions:
            break
        yield batch_questions


if __name__ == "__main__":
    args = arg_parser()

    # Do some initializations
    os.environ['HF_HOME'] = args.cache_dir
    os.environ['NLTK_DATA'] = args.cache_dir

    warnings.filterwarnings('ignore',
                            'MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization')

    langchain(args)
