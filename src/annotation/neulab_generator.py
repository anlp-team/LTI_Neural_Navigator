import json

from prompt2model import dataset_generator
from prompt2model.dataset_generator import PromptBasedDatasetGenerator, DatasetSplit
from prompt2model.prompt_parser import PromptBasedInstructionParser, TaskType, MockPromptSpec

# Set an API key as an environment variable. For instance, if using OpenAI:
#
# export OPENAI_API_KEY="<your-api-key>"

task_type = TaskType.TEXT_GENERATION
prompt_spec = MockPromptSpec(task_type)

instruction = """
You are a helpful assistant who can generate questions and answers based on a given document. 
You will be given a document and asked to generate a list of questions and answers based on the given document. 
The number of questions to generate will be provided as an input. 
The output will be a list of tuples, each containing a question and its answer.
"""  # A string indicating the task description.
examples = """
document: "The Los Angeles Dodgers won the World Series in 2020. The World Series in 2020 was played in Arlington, Texas. 
The Los Angeles Dodgers defeated the Tampa Bay Rays in six games. The Dodgers won the series for the first time since 1988."
num_questions: 2
output: [
    ("Who won the world series in 2020?", "The Los Angeles Dodgers won the World Series in 2020."),
    ("Where was it played?", "The World Series in 2020 was played in Arlington, Texas.")
]
"""  # A string indicating the examples.
prompt_spec._instruction = instruction
prompt_spec._examples = examples

textfile_path = "/Users/yuanye/Desktop/S24/11711/projects/LTI_Neural_Navigator/data/sample/html_samples/Carnegie_Mellon_University_-_Schedule_Of_Classes.txt"
with open(textfile_path, "r") as file:
    document_text = file.read()

prompt_dict = {
    "document": document_text
}
prompt = json.dumps(prompt_dict)
prompt_spec.parse_from_prompt(prompt)

generator = PromptBasedDatasetGenerator()

dataset = generator.generate_dataset_split(prompt_spec,
                                           num_examples=6,
                                           split=DatasetSplit.TRAIN)

dataset.save_to_disk("/Users/yuanye/Desktop/S24/11711/projects/LTI_Neural_Navigator/annotated_sample")
