# Project Overview
This project implements an advanced text retrieval and question-answering system utilizing state-of-the-art NLP techniques and models. The system is designed to process a query, retrieve relevant documents from a knowledge base, and generate accurate answers. It leverages the langchain ecosystem, Hugging Face transformers, and custom components for document embedding, retrieval, reranking, and question answering.

## Installation
Before running the project, ensure you have Python 3.8 or newer installed. It's recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```
Install the required dependencies:

```bash
pip install -r requirements.txt
```
The requirements.txt should include torch, transformers, sentence_transformers, tqdm, peft, and any other libraries used in main.py.

## Usage
The main.py script is designed to be run from the command line with various arguments to control its behavior. Here is how you can use it:

```bash
python main.py --knowledge_base <path_to_knowledge_base> --test_set <path_to_questions> --groundtruth <path_to_answers>
```

### Key Arguments
- --knowledge_base: The path to the directory containing your knowledge base documents.
- --test_set: The path to the file containing test set questions.
- --groundtruth: The path to the file containing the corresponding ground truth answers for the test set.
- --topk: The maximum number of documents to retrieve for answering a question.
- --device: The device to run the model on, e.g., cuda for GPU.
Refer to the script's argument parser section for a detailed explanation of all available options.

### Interactive Mode
To run the system in an interactive mode, where you can input questions directly and receive answers, simply omit the --test_set and --groundtruth arguments. Note: Interactive mode is marked as not implemented in the provided script, so some modifications might be needed to enable this feature.

## Components
The system integrates several key components:

- Embedding Model: Embeds input questions and documents into a high-dimensional space for efficient retrieval.
- Document Database: Stores and manages the knowledge base documents.
- Retriever: Retrieves the top-k most relevant documents based on the question embedding.
- Reranker: Further refines the relevance of the retrieved documents using a cross-encoder model.
- Question Answering Model: Generates answers based on the selected documents and the input question.

## Extending the System
This system is designed to be modular, allowing for easy swapping of components and integration of additional features, such as different models for embedding or reranking.
