import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="LTI Neural Navigator")
    parser.add_argument("--in_dir", type=str, default="/home/ubuntu/rag-project/data/2024-02-26/sample/",
                        help="Path to input directory")
    parser.add_argument("--out_dir", type=str, default="/home/ubuntu/rag-project/annotated_sample/",
                        help="Path to output directory")
    args = parser.parse_args()

    # Get a list of all files in the input directory
    input_dir = args.in_dir
    file_list = list_all_files(input_dir)

    # Get a list of files that have not been processed yet
    incremental_list = get_incremental_list(file_list, args)
    print_all_files(incremental_list)


def list_all_files(root_path):
    files_list = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            file_path = os.path.join(root, file)
            files_list.append(file_path)
    return files_list


def get_incremental_list(file_list, args):
    input_dir = args.in_dir
    output_dir = args.out_dir

    # Get a list of already processed files
    processed_files = list_all_files(output_dir)
    # Get relative file names relative to the output directory
    processed_files = [os.path.relpath(file, output_dir) for file in processed_files]
    # construct the full path of the processed files in the input directory
    processed_files = [os.path.join(input_dir, file) for file in processed_files]
    processed_files = [file.replace(".json", ".txt") for file in processed_files]

    # Filter out the files that have already been processed
    incremental_list = list(set(file_list) - set(processed_files))

    return incremental_list


def print_all_files(file_list):
    print("*" * 50)
    print("The following files will be processed:")
    for file in file_list:
        print(file)
    print("*" * 50)


if __name__ == "__main__":
    main()
