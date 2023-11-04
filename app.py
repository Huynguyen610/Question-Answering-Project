import os
import chainlit as cl
from langchain.llms import CTransformers
from langchain import PromptTemplate, LLMChain
from Question_Answering.pipeline.prediction import PredictionPipeline


local_llm = PredictionPipeline()

config = {
    "max_new_tokens": 1024,
    "repetition_penalty": 1.1,
    "temperature": 0.5,
    "top_k": 50,
    "top_p": 0.9,
    "stream": True,
    "threads": int(os.cpu_count() / 2)
}

template = """Question: {question}

Answer: Please refer to factual information and don't make up fictional data/information.
"""

@cl.on_chat_start
def main():
    prompt = PromptTemplate(template=template, input_variables=['question'])
    llm_chain = LLMChain(prompt=prompt, llm=local_llm, verbose=True)
    cl.user_session.set("llm_chain", llm_chain)


@cl.on_message
async def main(message: str):
    llm_chain = cl.user_session.get("llm_chain")
    res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])
    await cl.Message(content=res["text"]).send()


# chainlit run app.py
