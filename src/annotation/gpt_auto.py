import json

from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_questions_and_answers(document, num_questions):
    """
    Generates a list of questions and answers based on the given document.

    :param document: A string containing the document text.
    :param num_questions: The number of questions to generate.
    :return: A list of tuples, each containing a question and its answer.
    """
    questions_and_answers = []

    query_dict = {
        "document": document,
        "num_questions": num_questions
    }
    query_text = json.dumps(query_dict)

    try:
        # Generate a question
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": '''You are a helpful assistant who can generate questions and answers based on a given document. 
You will be given a document and asked to generate a list of questions and answers based on the given document. 
The number of questions to generate will be provided as an input. 
The output will be a list of tuples, each containing a question and its answer.'''},
                {"role": "user",
                 "content": '''document: The Los Angeles Dodgers won the World Series in 2020. The World Series in 2020 was played in Arlington, Texas. 
The Los Angeles Dodgers defeated the Tampa Bay Rays in six games. The Dodgers won the series for the first time since 1988."
num_questions: 2'''},
                {"role": "assistant",
                 "content": '''[
("Who won the world series in 2020?", "The Los Angeles Dodgers won the World Series in 2020."),
("Where was it played?", "The World Series in 2020 was played in Arlington, Texas.")
]'''},
                {"role": "user", "content": query_text}
            ]
        )
        response_text = response.choices[0].text.strip()
        questions_and_answers = json.loads(response_text)

    except Exception as e:
        print(f"An error occurred: {e}")

    return questions_and_answers


# Example usage
document_path = "../../data/sample/html_samples/A_history_of_SCS_|_SCS25_-_Carnegie_Mellon_University_School_of_Computer_Science.txt"
with open(document_path, "r") as file:
    document_text = file.read()

num_questions = 2  # Replace with your desired number of questions

qa_pairs = generate_questions_and_answers(document_text, num_questions)
for question, answer in qa_pairs:
    print(f"Q: {question}\nA: {answer}\n")
