from __future__ import annotations
from typing import Dict, Optional, Sequence
from langchain.schema import Document
from langchain.pydantic_v1 import Extra, root_validator

from langchain.callbacks.manager import Callbacks
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from sentence_transformers import CrossEncoder


# from config import bge_reranker_large


class BgeRerank(BaseDocumentCompressor):
    model_name: str = "avsolatorio/GIST-large-Embedding-v0"
    """Model name to use for reranking."""
    model: CrossEncoder = CrossEncoder(model_name)
    """CrossEncoder instance to use for reranking."""
    top_n: int = 10
    """Number of documents to return."""

    def bge_rerank(self, query, docs):
        model_inputs = [[query, doc] for doc in docs]
        scores = self.model.predict(model_inputs)
        results = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return results[: self.top_n]

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def compress_documents(
            self,
            documents: Sequence[Document],
            query: str,
            callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using BAAI/bge-reranker models.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        if len(documents) == 0:  # to avoid empty api call
            return []
        doc_list = list(documents)
        _docs = [d.page_content for d in doc_list]
        results = self.bge_rerank(query, _docs)
        final_results = []
        for r in results:
            doc = doc_list[r[0]]
            doc.metadata["relevance_score"] = r[1]
            final_results.append(doc)
        return final_results
