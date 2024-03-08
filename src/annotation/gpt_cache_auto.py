# pip install gptcache
# pip install "milvus[client]"

import argparse
import os
import time

from gptcache import cache
from gptcache.embedding import Onnx
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain.chat_models import ChatOpenAI, BedrockChat, AzureChatOpenAI, FakeListChatModel, PromptLayerChatOpenAI, \
    ChatDatabricks, ChatEverlyAI, ChatAnthropic, ChatCohere, ChatGooglePalm, ChatMlflow, ChatMLflowAIGateway, \
    ChatOllama, ChatVertexAI, JinaChat, HumanInputChatModel, MiniMaxChat, ChatAnyscale, ChatLiteLLM, ErnieBotChat, \
    ChatJavelinAIGateway, ChatKonko, PaiEasChatEndpoint, QianfanChatEndpoint, ChatFireworks, ChatYandexGPT, \
    ChatBaichuan, ChatHunyuan, GigaChat, VolcEngineMaasChat
from langchain.chains import QAGenerationChain
from gptcache.adapter.langchain_models import LangChainChat
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


def get_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model", type=str, default="ollama",
                            help="The model to use for generating questions and answers")
    arg_parser.add_argument("--local_input_dir",
                            type=str,
                            default="/home/ubuntu/data/2024-02-26/sample/",
                            help="Path to the original document")
    arg_parser.add_argument("--local_output_dir", type=str,
                            default="/home/ubuntu/data/test_out/",
                            help="Path to the output directory for the annotated document")
    arg_parser.add_argument("--num_qas", type=int, default=10, help="Number of questions and answers to generate")
    arg_parser.add_argument("--models_dir", type=str, default="/home/ubuntu/data/models/",
                            help="Path to the directory containing the models")
    arg_parser.add_argument("--use_s3", type=bool, default=False, help="Whether to use S3 for data storage")
    arg_parser.add_argument("--s3_input_dir", type=str, default="s3://lti-neural-navigator/data/raw/bs")
    arg_parser.add_argument("--s3_output_dir", type=str, default="s3://lti-neural-navigator/annotated_sample")
    arg_parser.add_argument("--local_tmp_dir", type=str, default="/home/ubuntu/data/tmp/")

    return arg_parser.parse_args()


# get the content(only question) form the prompt to cache
def get_msg_func(data, **_):
    return data.get("messages")[-1].content


def list_all_files(root_path):
    files_list = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            file_path = os.path.join(root, file)
            files_list.append(file_path)
    return files_list


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
    if args.model == "chat":
        cache.set_openai_key()

    text_splitter = RecursiveCharacterTextSplitter(chunk_overlap=500, chunk_size=2000)

    chat = LangChainChat(chat=get_model(args))

    if args.model == "chat":
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

        CHAT_PROMPT = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(templ1),
                HumanMessagePromptTemplate.from_template(templ2),
            ]
        )

        chain = QAGenerationChain.from_llm(chat, text_splitter=text_splitter, prompt=CHAT_PROMPT)
        filename_list = list_all_files(args.local_input_dir)
        for filename in filename_list:
            try:
                loader = TextLoader(filename)
                doc = loader.load()[0]
                qa = chain.run(doc.page_content)
                with open(f"{args.local_output_dir}/{args.file_name}", "w") as f:
                    f.write(qa)
            except Exception as e:
                print(e)
                print(f"An error occurred when trying to process file: {filename}")

    else:
        chain = QAGenerationChain.from_llm(chat, text_splitter=text_splitter)
        filename_list = list_all_files(args.local_input_dir)
        for filename in filename_list:
            # In order to be processed, the file must be a .txt file
            if not filename.endswith(".txt"):
                continue

            try:
                loader = TextLoader(filename)
                doc = loader.load()[0]
                qa = chain.run(doc.page_content)
                with open(f"{args.local_output_dir}/{args.file_name}", "w") as f:
                    f.write(qa)
            except Exception as e:
                print(e)
                print(f"An error occurred when trying to process file: {filename}")


def get_model(args):
    model_dict = {
        "chat": ChatOpenAI(temperature=0),
        "bedrock": BedrockChat,
        "azure": AzureChatOpenAI,
        "fake": FakeListChatModel,
        "prompt": PromptLayerChatOpenAI,
        "databricks": ChatDatabricks,
        "everly": ChatEverlyAI,
        "anthropic": ChatAnthropic,
        "cohere": ChatCohere,
        "google": ChatGooglePalm,
        "mlflow": ChatMlflow,
        "mlflowai": ChatMLflowAIGateway,
        "ollama": ChatOllama,
        "vertex": ChatVertexAI,
        "jina": JinaChat,
        "human": HumanInputChatModel,
        "minimax": MiniMaxChat,
        "anyscale": ChatAnyscale,
        "lite": ChatLiteLLM,
        "ernie": ErnieBotChat,
        "javelin": ChatJavelinAIGateway,
        "konko": ChatKonko,
        "pai": PaiEasChatEndpoint,
        "qianfan": QianfanChatEndpoint,
        "fireworks": ChatFireworks,
        "yandex": ChatYandexGPT,
        "baichuan": ChatBaichuan,
        "hunyuan": ChatHunyuan,
        "giga": GigaChat,
        "volcengine": VolcEngineMaasChat
    }

    return model_dict[args.model]


if __name__ == "__main__":
    args = get_args()

    start_time = time.time()

    args.file_name = args.local_input_dir.split("/")[-1]

    # download the document from S3
    if args.use_s3:
        args.local_input_dir = os.path.join(args.local_tmp_dir, "input")
        args.local_output_dir = os.path.join(args.local_tmp_dir, "output")

        if not os.path.exists(args.local_input_dir):
            os.makedirs(args.local_input_dir)
        if not os.path.exists(args.local_output_dir):
            os.makedirs(args.local_output_dir)

        # recursively download the directory from S3
        os.system(f"aws s3 cp --recursive {args.s3_input_dir} {args.local_input_dir}")

    main(args)

    # upload the annotated document to S3
    if args.use_s3:
        os.system(f"aws s3 cp --recursive {args.local_output_dir} {args.s3_output_dir}")
        # clean up the local tmp directory
        # os.system(f"rm -rf {args.local_tmp_dir}")  # not safe, don't do this :(

    print("Done!")
    print(f"Time: {time.time() - start_time:.2f}")
