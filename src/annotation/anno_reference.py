import argparse
import json
import os
import time

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


def arg_parser():
    parser = argparse.ArgumentParser(description='Annotate reference genome')
    parser.add_argument("--embed_model_id", type=str,
                        # default="intfloat/e5-mistral-7b-instruct",
                        default="mixedbread-ai/mxbai-embed-large-v1",
                        help="Model ID for the embedding model")
    parser.add_argument("--chunk_size", type=int, default=500, help="Chunk size")
    parser.add_argument("--chunk_overlap", type=int, default=150, help="Chunk overlap")
    parser.add_argument("--topk", type=int, default=5, help="Top k documents to retrieve")
    parser.add_argument("--src_path", type=str,
                        default="/home/ubuntu/rag-project/annotated_sample/")
    parser.add_argument("--dst_path", type=str,
                        default="/home/ubuntu/rag-project/dataset_with_ref/")
    parser.add_argument("--incremental", type=bool, default=True,
                        help="Whether to process only the files that have not been processed yet.")

    parser.add_argument("--cache_dir", type=str, default="/mnt/datavol/cache",
                        help="Directory to store cache")
    parser.add_argument("--search_type", type=str, default="mmr", help="Type of search to perform",
                        choices=["similarity", "mmr", "similarity_score_threshold"])
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")

    args = parser.parse_args()
    return args


def annotation(args):
    todo_files = list_all_files(args.src_path)
    if args.incremental:
        todo_files = get_incremental_list(todo_files, args)

    # debug
    todo_files = ["/home/ubuntu/rag-project/annotated_sample/gpt4all-graham.json"]

    # sort files by file size
    todo_files.sort(key=os.path.getsize)
    print_all_files(todo_files)

    for file in todo_files:
        print(f"Processing {file} ...")
        start_time = time.time()

        # Process the file
        with open(file, 'r') as f:
            content = f.read()

        src_dict = json.loads(content)
        doc_text = src_dict['doc_text']
        dst_dict = {
            "file_path": src_dict['file_path'],
            "num_qa_pairs": src_dict['num_qa_pairs'],
            "doc_text": doc_text,
            "qa_list": []
        }

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )
        all_splits = text_splitter.create_documents([doc_text])

        embeddings = HuggingFaceEmbeddings(
            model_name=args.embed_model_id,
            model_kwargs={
                "device": args.device,
            },
            encode_kwargs={

            },
            cache_folder=args.cache_dir
        )

        vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)

        for qa_pair in src_dict['qa_list']:
            question = qa_pair['question']
            try:
                top_k_docs = vectorstore.search(question, k=args.topk, search_type=args.search_type)
                dst_qa_pair = {
                    "question": question,
                    "answer": qa_pair['answer'],
                    "top_k_docs": {i: doc.page_content for i, doc in enumerate(top_k_docs)}
                }
                dst_dict['qa_list'].append(dst_qa_pair)
            except Exception as e:
                print(f"Error processing question: {question}")
                print(e)

        # Save the processed file
        dst_file = get_dst_filename(file, args.src_path, args.dst_path)
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        with open(dst_file, 'w') as f:
            json.dump(dst_dict, f, indent=4)

        print(f"Processed {file} in {time.time() - start_time} seconds.")
        print(f"Saved to {dst_file}")
        print("*" * 50)

        # cleanup the vector store
        vectorstore.delete_collection()


def list_all_files(root_path):
    files_list = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            file_path = os.path.join(root, file)
            files_list.append(file_path)
    return files_list


def get_incremental_list(file_list, args):
    input_dir = args.src_path
    output_dir = args.dst_path

    # Get a list of already processed files
    processed_files = list_all_files(output_dir)
    # Get relative file names relative to the output directory
    processed_files = [os.path.relpath(file, output_dir) for file in processed_files]
    # construct the full path of the processed files in the input directory
    processed_files = [os.path.join(input_dir, file) for file in processed_files]

    # Filter out the files that have already been processed
    incremental_list = list(set(file_list) - set(processed_files))

    return incremental_list


def get_dst_filename(src_file, src_path, dst_path):
    src_file = os.path.relpath(src_file, src_path)
    return os.path.join(dst_path, src_file)


def print_all_files(file_list):
    print("*" * 50)
    print(f"The following files will be processed:")
    for file in file_list:
        print(file)
    print(f"Summary: {len(file_list)} files in total.")
    print("*" * 50)


def main():
    args = arg_parser()
    annotation(args)


if __name__ == "__main__":
    main()
