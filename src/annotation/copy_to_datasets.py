import argparse
import json
import os
import shutil


def arg_parser():
    args = argparse.ArgumentParser(description="Copy files from one directory to another.")
    args.add_argument("--src_path", type=str,
                      default="/Users/yuanye/Desktop/S24/11711/projects/LTI_Neural_Navigator/annotated_sample/",
                      help="Source directory")
    args.add_argument("--dst_path", type=str,
                      default="/Users/yuanye/Desktop/S24/11711/projects/LTI_Neural_Navigator/dataset_ref_chunk/",
                      help="Destination directory")

    return args.parse_args()


def main_worker(args):
    all_files = list_all_files(args.src_path)
    # filter out new format files
    new_format_files = []
    for file in all_files:
        # load the file with json
        with open(file, "r") as f:
            try:
                jsondict = json.load(f)
                if (len(jsondict["qa_list"]) > 0
                        and "ref_chunk" in jsondict["qa_list"][0]):
                    new_format_files.append(file)
            except Exception as e:
                print(f"Error loading file {file}: {e}")
                exit(1)

    # get incremental list
    incremental_list = get_incremental_list(new_format_files, args)
    print_all_files(incremental_list)

    # copy files
    for file in incremental_list:
        print(f"Processing {file} ...")
        dst_file = get_dst_filename(file, args.src_path, args.dst_path)
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        shutil.copy(file, dst_file)

    print("All files have been processed.")


def list_all_files(root_path):
    files_list = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if not file.endswith(".json"):
                continue
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
    main_worker(args)


if __name__ == "__main__":
    main()
