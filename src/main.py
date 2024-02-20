import argparse

from langchain import hub
from langchain_community.llms import GPT4All

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from rag.doc_reader import QuestionAnsweringModel, ContextSelector, AnswerProcessor
from rag.embedder import EmbeddingModel, Embedder
from rag.retriever import DocumentDatabase, Ranker, Retriever



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
    loader = DirectoryLoader(args.doc_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)

    vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())

    # prepare core model
    gpt4all = GPT4All(model=args.model_ckpt, max_tokens=args.max_tokens)

    # Prompt
    rag_prompt = hub.pull("rlm/rag-prompt")

    # Chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    retriever = vectorstore.as_retriever()
    chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | rag_prompt
            | gpt4all
            | StrOutputParser()
    )

    # Run
    invoke_result = chain.invoke(args.question)
    print(invoke_result)


def arg_parser():
    parser = argparse.ArgumentParser(description="LTI Neural Navigator")
    parser.add_argument("--question", type=str,
                        default="What are the approaches to Task Decomposition?",
                        help="Question to ask")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum tokens supported by the system")

    parser.add_argument("--doc_path", type=str, default="../data/sample", help="Path to knowledge base")
    parser.add_argument("--core_model_ckpt", type=str, default="../model_ckpts/????", help="Path to core model checkpoint")
    parser.add_argument("--embed_model_ckpt", type=str, default="../model_ckpts/????", help="Path to model checkpoint")

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    langchain(args)
