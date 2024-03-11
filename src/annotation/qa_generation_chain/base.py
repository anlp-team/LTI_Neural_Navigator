from __future__ import annotations

import json
import json_repair
from json_repair import repair_json
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import Field
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.qa_generation.prompt import PROMPT_SELECTOR


class QAGenerationChain(Chain):
    """Base class for question-answer generation chains."""

    llm_chain: LLMChain
    """LLM Chain that generates responses from user input and context."""
    model_name: str
    """Name of the language model."""
    text_splitter: TextSplitter = Field(
        default=RecursiveCharacterTextSplitter(chunk_overlap=500)
    )
    """Text splitter that splits the input into chunks."""
    input_key: str = "text"
    """Key of the input to the chain."""
    output_key: str = "qa_pairs"
    """Key of the output of the chain."""
    k: Optional[int] = None
    """Number of questions to generate."""

    @classmethod
    def from_llm(
            cls,
            llm: BaseLanguageModel,
            prompt: Optional[BasePromptTemplate] = None,
            **kwargs: Any,
    ) -> QAGenerationChain:
        """
        Create a QAGenerationChain from a language model.

        Args:
            llm: a language model
            prompt: a prompt template
            **kwargs: additional arguments

        Returns:
            a QAGenerationChain class
        """
        _prompt = prompt or PROMPT_SELECTOR.get_prompt(llm)
        chain = LLMChain(llm=llm, prompt=_prompt)
        return cls(llm_chain=chain, **kwargs)

    @property
    def _chain_type(self) -> str:
        raise NotImplementedError

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def normalize_dirty_str(self, dirty_str: str) -> str:
        if self.model_name == "chat":
            raise NotImplementedError
        elif self.model_name == "wizardlm":
            if "```" in dirty_str:
                s = dirty_str.split("```")[1].strip()
                if s.startswith("json"):
                    s = s[len("json"):].strip()
            else:
                s = dirty_str[len("text="):][1:-1].strip()
            return (s.replace("\\'", "'")
                    .replace("\\n", "\n")
                    .strip())
        elif self.model_name == "llama2":
            s = dirty_str[len("text="):][1:-1].strip()
            return (s.replace("\\'", "'")
                    .replace("\\n", "\n")
                    .strip())
        elif self.model_name == "gpt4all":
            # gpt4all is really naughty
            if "```" in dirty_str:
                s = dirty_str.split("```")[1]
            else:
                s = dirty_str[len("text="):][1:-1].strip()
            return (s.replace("\\'", "'")
                    .replace("\\n", "\n")
                    .strip())
        else:
            raise NotImplementedError

    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        docs = self.text_splitter.create_documents([inputs[self.input_key]])
        results = self.llm_chain.generate(
            [{"text": d.page_content} for d in docs], run_manager=run_manager
        )
        qa_ret = []
        for idx, res in enumerate(results.generations):
            dirty_str = str(res[0])
            try:
                json_str = self.normalize_dirty_str(dirty_str)
                qa_list = json.loads(json_str)
                for qa in qa_list:
                    qa["ref_chunk"] = docs[idx].page_content
                qa_ret.extend(qa_list)
            except NotImplementedError as e:
                raise e
            except Exception as e:
                try:
                    bad_json_str = self.normalize_dirty_str(dirty_str)
                    qa_list = json_repair.loads(bad_json_str)
                    if not isinstance(qa_list, list):
                        raise ValueError(f"Json repairs failed, expected a list, but got {type(qa_list)}")
                    for qa in qa_list:
                        qa["ref_chunk"] = docs[idx].page_content
                    qa_ret.extend(qa_list)
                except Exception as e:
                    print(f"Json cannot decode the string: {dirty_str}")
                    print(f"Error: {e}")

        return {self.output_key: qa_ret}
