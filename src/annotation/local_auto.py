import argparse
import json
import os
import time
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import GPT4All
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, \
    PromptTemplate

from qa_generation_chain import QAGenerationChain


def get_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--device", type=str, default="NVIDIA A10G",
                            choices=["cpu", "gpu"],
                            help="Device to use for the model")
    arg_parser.add_argument("--model", type=str, default="GPT4All",
                            help="The model to use for generating questions and answers")
    arg_parser.add_argument("--model_path", type=str,
                            # default="/home/ubuntu/data/models/nous-hermes-llama2-13b.Q4_0.gguf",
                            # default="/home/ubuntu/data/models/wizardlm-13b-v1.2.Q4_0.gguf",
                            default="/home/ubuntu/data/models/gpt4all-13b-snoozy-q4_0.gguf",
                            help="Path to the model")
    arg_parser.add_argument("--max_tok", type=int, default=2048, help="Maximum number of tokens in the model")
    arg_parser.add_argument("--local_input_dir",
                            type=str,
                            default="/home/ubuntu/rag-project/data/2024-02-26/sample/",
                            help="Path to the original document")
    arg_parser.add_argument("--local_output_dir", type=str,
                            default="/home/ubuntu/data/test_out/",
                            help="Path to the output directory for the annotated document")
    arg_parser.add_argument("--num_qas", type=int, default=10,
                            help="Number of questions and answers to generate for each chunk "
                                 "(a long document may be split into multiple chunks)")
    arg_parser.add_argument("--models_dir", type=str, default="/home/ubuntu/data/models/",
                            help="Path to the directory containing the models")
    arg_parser.add_argument("--use_s3", type=bool, default=False, help="Whether to use S3 for data storage")
    arg_parser.add_argument("--s3_input_dir", type=str, default="s3://lti-neural-navigator/data/raw/bs")
    arg_parser.add_argument("--s3_output_dir", type=str, default="s3://lti-neural-navigator/annotated_sample")
    arg_parser.add_argument("--local_tmp_dir", type=str, default="/home/ubuntu/data/tmp/")

    arg_parser.add_argument("--debug", type=bool, default=True, help="Whether to print debug information")
    arg_parser.add_argument("--select_file_list", type=list,
                            # default=None,
                            default=[
                                "/home/ubuntu/rag-project/data/2024-02-26/sample/Graham_Neubig_|_Carnegie_Mellon_University_-_Language_Technologies_Institute.txt"],
                            help="List of files to process, only for debug")

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
    text_splitter = RecursiveCharacterTextSplitter(chunk_overlap=500, chunk_size=2000)

    model = get_model(args)

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

        chain = QAGenerationChain.from_llm(model,
                                           model_name="chat",
                                           text_splitter=text_splitter,
                                           prompt=CHAT_PROMPT)
        if args.select_file_list is not None:
            filename_list = args.select_file_list
        else:
            filename_list = list_all_files(args.local_input_dir)

        for filename in filename_list:
            try:
                loader = TextLoader(filename)
                doc = loader.load()[0]
                qa = chain.run(doc.page_content)
                with open(os.path.join(args.local_output_dir, args.file_name), "w") as f:
                    f.write(qa)
            except Exception as e:
                print(e)
                print(f"An error occurred when trying to process file: {filename}")

    else:
        # local models

        # prepare prompt template
        # todo: extract the template to a separate file
        if args.model_name == "gpt4all":
            templ = f"""
            ### Instruction:
            You are a smart assistant designed to help high school teachers come up with reading comprehension questions.
            Given a piece of text, you must come up with {args.num_qas} question and answer pairs that can be used to test a student's reading comprehension abilities.
            When coming up with question/answer pairs, you must respond in the following format:
            ```
            [
                {{{{
                    "question": "YOUR_QUESTION_HERE",
                    "answer": "THE_ANSWER_HERE"
                }}}},
                {{{{
                    "question": "YOUR_SECOND_QUESTION_HERE",
                    "answer": "THE_SECOND_ANSWER_HERE"
                }}}}
            ]
            ```
            Everything between the ``` must be valid array.
    
            Please come up with {args.num_qas} question/answer pairs, in the specified JSON format, for the following text:
            ----------------
            {{text}}
            ### Response:
            """
            PROMPT = PromptTemplate.from_template(templ)
        else:
            templ = f"""
            ### Instruction:
            You are a smart assistant designed to help high school teachers come up with reading comprehension questions.
            Given a piece of text, you must come up with {args.num_qas} question and answer pairs that can be used to test a student's reading comprehension abilities.
            When coming up with question/answer pairs, you must respond in the following format:
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
    
            Please come up with {args.num_qas} question/answer pairs, in the specified JSON format, for the following text:
            ----------------
            {{text}}
            ### Response:
            """
            PROMPT = PromptTemplate.from_template(templ)

        chain = QAGenerationChain.from_llm(model,
                                           model_name=args.model_name,
                                           text_splitter=text_splitter,
                                           prompt=PROMPT)
        if args.select_file_list is not None:
            filename_list = args.select_file_list
        else:
            filename_list = list_all_files(args.local_input_dir)

        for filename in filename_list:
            # In order to be processed, the file must be a .txt file
            if not filename.endswith(".txt"):
                continue
            print(f"Processing file: {filename}")

            with open(filename, "r") as file:
                document_text = file.read()

            try:
                loader = TextLoader(str(filename))
                doc = loader.load()[0]
                qa = chain.invoke({"text": doc.page_content})
                qa_pairs = qa["qa_pairs"]
                normalized_filename = filename.split("/")[-1].replace(".txt", ".json")
                with open(os.path.join(args.local_output_dir, normalized_filename), "w") as f:
                    output_dict = {
                        "file_path": filename,
                        "num_qa_pairs": len(qa_pairs),
                        "qa_list": qa_pairs,
                        "doc_text": document_text
                    }
                    f.write(json.dumps(output_dict))
                print('*' * 50)
                print(f"For file: {filename}, the questions and answers are:\n\n {qa_pairs}")
                print('*' * 50)
            except Exception as e:
                print(e)
                print(f"An error occurred when trying to process file: {filename}")


def get_model(args):
    # https://python.langchain.com/docs/guides/local_llms
    if args.model == "LlamaCpp":
        raise NotImplementedError("LlamaCpp is not supported yet")
        args.model_name = "llama2"
    elif args.model == "GPT4All":
        print(f"Using GPT4All, loading model from: {args.model_path}")

        # set args.model_name
        if "gpt4all" in args.model_path:
            args.model_name = "gpt4all"
        elif "wizardlm" in args.model_path:
            args.model_name = "wizardlm"
        elif "llama2" in args.model_path:
            args.model_name = "llama2"

        return GPT4All(
            model=args.model_path,
            max_tokens=args.max_tok,
            device=args.device,
            n_batch=2048,
            n_predict=2048,
            seed=-1,
            verbose=args.debug,
        )
    elif args.model == "chat":
        print(f"Using ChatGPT online model.")
        args.model_name = "chat"
        return ChatOpenAI(temperature=0)
    else:
        raise ValueError(f"Invalid model: {args.model}")


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
