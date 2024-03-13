# Team Contributions

## Data Collection and Preprocessing Contributions

- **Jiarui Li**: Led the data collection phase by writing scripts for scraping HTML pages, implementing preprocessing steps to convert HTML and pdf to plain text format. Focused on removing noise, redundant information, and deduplication within the collected datasets. 
- **Ye Yuan**: Was responsible for setting up Langchain with different pretrained LLMs, developing a sample RAG system. Contributed to the evaluation of the system using metrics such as answer recall, exact match, and F1. Investigated other possible models that could be integrated into the project for improved performance.
- **Zehua Zhang**: Focused on the preprocessing of PDF documents using tools like `pypdf` and `pdfplumber`, and plaintext processing. Organized data collected from Semantic Scholar API by Professor & Field, ensuring the data's hierarchy and relevance to the project's needs were maintained. Convert all paper into text formats.

## Data Annotation Contributions

- **Jiarui Li**: Selected representative data points. Annotated html data, focusing on the creation of the auto-annotator for zero/few-shot question generation at both paragraph and document levels. 
- **Ye Yuan**: Implemented auto-annotation. Annotated pdf data, responsible for the overall annotation quality evaluation, including setting up metrics for inter-annotator agreement and leading efforts to evaluate the annotation process's efficiency and accuracy.
- **Zehua Zhang**: Gather and organized paper data. Annotated paper data, working on the Semantic Scholar API to organize data by Professor & Field, which involved manual curation and validation to ensure the quality of data annotations.

## Modeling Contributions

- **Jiarui Li**: Focused on embedding model finetuning, working with tools and frameworks like Qlora & DeepSpeed Zero to finetune Llama2. Implemented a finetuning pipeline and contributed to the development of a custom reranker to optimize the retrieval process within the RAG system.
- **Ye Yuan**: Finalized the RAG pipeline, ensuring the system's components worked seamlessly together. Contributed to the auto-annotation process, making it highly scalable, and played a pivotal role in creating an AMI for easy deployment and replication of the project's environment.
- **Zehua Zhang**: Led the effort on generative model selection and finetuning, assessing the performance of different models and their fit for the project's goals. Contributed to the reranking module, designing a system to reorder document chunks based on relevance.

## Other Contributions

All team members were involved in various phases of the project, from initial planning and task distribution to the final phases of tuning the system towards the test set, writing the report, and organizing the repository for submission. Regular meetings ensured that tasks were distributed evenly, and progress was consistently monitored and adjusted as needed.


