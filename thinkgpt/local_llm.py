from typing import List, Optional, Dict, Any, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.llms import (
    BaseLLM,
    HuggingFaceTextGenInference,
    HuggingFaceHub,
    HuggingFaceEndpoint,
)
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.schema import LLMResult, Generation
from docarray import DocumentArray
from pydantic.config import Extra

from thinkgpt.abstract import AbstractMixin, AbstractChain
from thinkgpt.condition import ConditionMixin, ConditionChain
from thinkgpt.infer import InferMixin, InferChain
from thinkgpt.memory import MemoryMixin, ExecuteWithContextChain
from thinkgpt.refine import RefineMixin, RefineChain
from thinkgpt.gpt_select import SelectChain, SelectMixin

from thinkgpt.summarize import SummarizeMixin, SummarizeChain


class ThinkGPT(
    ChatHuggingFace,
    MemoryMixin,
    AbstractMixin,
    RefineMixin,
    ConditionMixin,
    SelectMixin,
    InferMixin,
    SummarizeMixin,
    extra=Extra.allow,
):
    """Wrapper around HuggingFace large language models to augment it with memory

    To use, you should have the ``langchain_community`` python package installed.
    """

    def __init__(
        self,
        llm: Union[HuggingFaceTextGenInference, HuggingFaceEndpoint, HuggingFaceHub],
        memory: DocumentArray = None,
        execute_with_context_chain: ExecuteWithContextChain = None,
        abstract_chain: AbstractChain = None,
        refine_chain: RefineChain = None,
        condition_chain: ConditionChain = None,
        select_chain: SelectChain = None,
        infer_chain: InferChain = None,
        summarize_chain: SummarizeChain = None,
        verbose=True,
        embedding_kwargs: Dict[str, Any] = None,
        # TODO: model name can be specified per mixin
        **kwargs
    ):
        super().__init__(llm=llm, **kwargs)
        # TODO: offer more docarray backends
        self.memory = memory or DocumentArray()
        self.embeddings_model = HuggingFaceEmbeddings(**(embedding_kwargs or {}))
        # self.llm = llm
        self.chat_model = ChatHuggingFace(llm=llm, **kwargs)
        self.execute_with_context_chain = (
            execute_with_context_chain
            or ExecuteWithContextChain(llm=self.chat_model, verbose=verbose)
        )
        self.abstract_chain = abstract_chain or AbstractChain(
            llm=self.chat_model, verbose=verbose
        )
        self.refine_chain = refine_chain or RefineChain(
            llm=self.chat_model, verbose=verbose
        )
        self.condition_chain = condition_chain or ConditionChain(
            llm=self.chat_model, verbose=verbose
        )
        self.select_chain = select_chain or SelectChain(
            llm=self.chat_model, verbose=verbose
        )
        self.infer_chain = infer_chain or InferChain(
            llm=self.chat_model, verbose=verbose
        )
        self.summarize_chain = summarize_chain or SummarizeChain(
            llm=self.chat_model, verbose=verbose
        )  # Add this line
        self.mem_cnt = 0

    def generate(
        self,
        prompts: List[List[str]],
        stop: Optional[List[str]] = None,
        remember: Union[int, List[str]] = 0,
    ) -> LLMResult:
        # only works for single prompt
        if len(prompts) > 1:
            raise Exception("only works for a single prompt")
        prompt = prompts[0][0]
        if isinstance(remember, int) and remember > 0:
            remembered_elements = self.remember(prompt, limit=5)
            result = self.execute_with_context_chain.predict(
                prompt=prompt,
                context=(
                    "\n".join(remembered_elements) if remembered_elements else "Nothing"
                ),
            )
        elif isinstance(remember, list):
            result = self.execute_with_context_chain.predict(
                prompt=prompt, context="\n".join(remember)
            )
        else:
            result = self.execute_with_context_chain.predict(
                prompt=prompt, context="Nothing"
            )

        return LLMResult(generations=[[Generation(text=result)]])

    def predict(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        remember: Union[int, List[str]] = 0,
    ) -> str:
        return self.generate([[prompt]], remember=remember).generations[0][0].text


if __name__ == "__main__":

    base_llm = HuggingFaceTextGenInference(
        inference_server_url="http://192.168.1.20:1315/",
        max_new_tokens=4096,
        top_k=10,
        top_p=0.95,
        typical_p=0.95,
        temperature=0.01,
        repetition_penalty=1.03,
        stop_sequences=["</s>", "[/INST]"],
    )

    print("Base llm:", base_llm)

    llm = ThinkGPT(
        llm=base_llm, model_id="mistralai/Mixtral-8x7B-Instruct-v0.1", verbose=True
    )

    rules = llm.abstract(
        observations=[
            'in tunisian, I did not eat is "ma khditech"',
            'I did not work is "ma khdemtech"',
            'I did not go is "ma mchitech"',
        ],
        instruction_hint="output the rule in french",
    )
    llm.memorize(rules)

    llm.memorize('in tunisian, I went is "mchit"')

    task = "translate to Tunisian: I didn't go"
    print(llm.predict(task, remember=llm.remember(task)))
