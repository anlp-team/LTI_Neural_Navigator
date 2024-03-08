# pip install gptcache
# pip install "milvus[client]"

import argparse
import os

from gptcache import cache
from gptcache.embedding import Onnx
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import QAGenerationChain
from gptcache.adapter.langchain_models import LangChainChat
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


# get the content(only question) form the prompt to cache
def get_msg_func(data, **_):
    return data.get("messages")[-1].content


def main(args):
    onnx = Onnx()
    cache_base = CacheBase('sqlite')
    vector_base = VectorBase("faiss", dimension=onnx.dimension)
    data_manager = get_data_manager(cache_base, vector_base)
    cache.init(
        pre_embedding_func=get_msg_func,
        embedding_func=onnx.to_embeddings,
        data_manager=data_manager,
        similarity_evaluation=SearchDistanceEvaluation(),
    )
    cache.set_openai_key()

    text_splitter = RecursiveCharacterTextSplitter(chunk_overlap=500, chunk_size=2000)
    loader = TextLoader(args.doc_path)
    doc = loader.load()[0]

    chat = LangChainChat(chat=ChatOpenAI(temperature=0))

    # thanks: https://www.reddit.com/r/LangChain/comments/12pwu0o/questions_about_qagenerationchain/
    templ1 = f"""You are a smart assistant designed to help high school teachers come up with reading comprehension questions.
    Given a piece of text, you must come up with {args.num_qas} question and answer pairs that can be used to test a student's reading comprehension abilities.
    When coming up with this question/answer pair, you must respond in the following format:
    ```
    [
    {{{{
        "question": "$YOUR_QUESTION_HERE",
        "answer": "$THE_ANSWER_HERE"
    }}}},
    {{{{
        "question": "$YOUR_SECOND_QUESTION_HERE",
        "answer": "$THE_SECOND_ANSWER_HERE"
    }}}}
    ]
    ```
    Everything between the ``` must be valid array.
    """
    templ2 = f"""Please come up with {args.num_qas} question/answer pairs, in the specified format, for the following text:
    ----------------
    {{text}}"""

    # debug
    print(templ1)
    print(templ2)

    CHAT_PROMPT = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(templ1),
            HumanMessagePromptTemplate.from_template(templ2),
        ]
    )

    chain = QAGenerationChain.from_llm(chat, text_splitter=text_splitter, prompt=CHAT_PROMPT)

    qa = chain.run(doc.page_content)

    with open(f"{args.qa_output_dir}/{args.file_name}", "w") as f:
        f.write(qa)

    print(qa)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--doc_path",
                      type=str,
                      default="/Users/yuanye/Desktop/S24/11711/projects/LTI_Neural_Navigator/data/2024-02-26/sample/________Index_____-__Carnegie_Mellon_University_Athletics.txt",
                      help="Path to the original document")
    args.add_argument("--qa_output_dir", type=str,
                      default="/Users/yuanye/Desktop/S24/11711/projects/LTI_Neural_Navigator/annotated_sample",
                      help="Path to the output directory for the annotated document")
    args.add_argument("--num_qas", type=int, default=6, help="Number of questions and answers to generate")

    args = args.parse_args()

    args.file_name = args.doc_path.split("/")[-1]

    main(args)
