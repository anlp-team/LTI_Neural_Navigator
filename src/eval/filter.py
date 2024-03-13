def filer_out_keywords_from_file(file_path, keywords, output_file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    filtered_lines = list()
    cnt = 0
    for line in lines:
        for keyword in keywords:
            if keyword in line:
                # replace the keyword with an empty string
                line = line.replace(keyword, "")
                cnt += 1
        filtered_lines.append(line)

    with open(output_file_path, "w") as file:
        file.writelines(filtered_lines)

    print(
        f"Filtered out {cnt} lines from {file_path} and saved the result to {output_file_path}"
    )


def main():
    keywords = [
        "Based on the provided context, ",
        "Based on the context provided, ",
        "According to the context provided, ",
        "Based on the provided context",
        "Based on the context provided",
        "According to the context provided",
    ]
    file_path = "/home/ubuntu/LTI_Neural_Navigator/testset/final_test_answer_raw.txt"
    output_file_path = (
        "/home/ubuntu/LTI_Neural_Navigator/testset/final_test_answer_raw_filtered.txt"
    )
    filer_out_keywords_from_file(file_path, keywords, output_file_path)


if __name__ == "__main__":
    main()
