import os
import json
import random


def list_all_files(root_path):
    files_list = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if not file.endswith(".json"):
                continue
            file_path = os.path.join(root, file)
            files_list.append(file_path)
    return files_list


def load_qa_pairs(dir_path, write_fp_q, write_fp_a):
    count_invalid = {
        "no_qa_pairs": 0,
        "no_question_answer": 0,
        "empty_answer_list": 0,
        "empty_answer": 0,
        "newline_in_qa": 0,
    }
    files_list = list_all_files(dir_path)
    for file in files_list:
        with open(file, "r") as f:
            data = json.load(f)
            qa_pairs = data["qa_list"]

            if len(qa_pairs) == 0:
                print(f"File {file} has no qa pairs")
                count_invalid["no_qa_pairs"] += 1
                continue

            for qa_pair in qa_pairs:
                if "question" not in qa_pair or "answer" not in qa_pair:
                    print(f"File {file} has a qa pair without question or answer")
                    count_invalid["no_question_answer"] += 1
                    continue
                question = qa_pair["question"]
                answer = qa_pair["answer"]
                
                # if answer is a list, convert it to a string
                if isinstance(answer, list):
                    if len(answer) == 0:
                        print(f"File {file} has a qa pair with empty answer list")
                        count_invalid["empty_answer_list"] += 1
                        continue
                    answer = "; ".join([str(ans) for ans in answer])

                if len(question) == 0 or len(answer) == 0:
                    print(f"File {file} question or answer is empty")
                    count_invalid["empty_answer"] += 1
                    continue

                if '\n' in question or '\n' in answer:
                    print(f"File {file} question or answer contains newline")
                    count_invalid["newline_in_qa"] += 1
                    question = question.replace('\n', '; ')
                    answer = answer.replace('\n', '; ')

                assert question[-1] != "\n" and answer[-1] != "\n"
                write_fp_q.write(question + "\n")
                write_fp_a.write(answer + "\n")

    print(f"Invalid qa pairs: {count_invalid}")


def train_test_split(root_path):
    quesiton_fp = open(os.path.join(root_path, "questions.txt"), "r")
    answer_fp = open(os.path.join(root_path, "answers.txt"), "r")
    question_lines = quesiton_fp.readlines()
    answer_lines = answer_fp.readlines()
    print(f"Number of questions: {len(question_lines)}")
    print(f"Number of answers: {len(answer_lines)}")
    assert len(question_lines) == len(answer_lines)

    # randomize the order of the lines
    combined = list(zip(question_lines, answer_lines))
    random.shuffle(combined)
    question_lines[:], answer_lines[:] = zip(*combined)
    # split the data into train and test
    train_q_fp = open(os.path.join(root_path, "train_questions.txt"), "w")
    train_a_fp = open(os.path.join(root_path, "train_answers.txt"), "w")
    test_q_fp = open(os.path.join(root_path, "test_questions.txt"), "w")
    test_a_fp = open(os.path.join(root_path, "test_answers.txt"), "w")
    split = int(0.8 * len(question_lines))
    train_q_fp.writelines(question_lines[:split])
    train_a_fp.writelines(answer_lines[:split])
    test_q_fp.writelines(question_lines[split:])
    test_a_fp.writelines(answer_lines[split:])
    quesiton_fp.close()
    answer_fp.close()
    train_q_fp.close()
    train_a_fp.close()
    test_q_fp.close()
    test_a_fp.close()

    print(f"Train questions: {len(question_lines[:split])}")
    print(f"Train answers: {len(answer_lines[:split])}")
    print(f"Test questions: {len(question_lines[split:])}")
    print(f"Test answers: {len(answer_lines[split:])}")

def main():
    json_data_paths = ["./dataset_ref_chunk/", "./embedder_dataset/"]
    write_fp_q = open("./data/QAparallel/questions.txt", "w")
    write_fp_a = open("./data/QAparallel/answers.txt", "w")
    for json_data_path in json_data_paths:
        load_qa_pairs(json_data_path, write_fp_q, write_fp_a)
    write_fp_q.close()
    write_fp_a.close()

    train_test_split("./data/QAparallel")


if __name__ == "__main__":
    main()
