import os
import chainlit as cl
from chainlit import Text
from Question_Answering.pipeline.prediction import PredictionPipeline

# chainlit run app.py -w to run


def make_prediction(context, question):
    prediction = PredictionPipeline()
    output = prediction.predict(context, question)
    return output


@cl.on_chat_start
async def start():
    res = await cl.AskUserMessage(content="What is your name?", timeout=30).send()
    if res:
        await cl.Message(
            content=f"Hi {res['content']}.\nI am an Extractive Question-Answering Bot!\nPlease type \'next\' to continue"
        ).send()


@cl.on_message
async def main(message: cl.Message):
    # ask for user input
    context_res = await cl.AskUserMessage(
        content="Please provide your Text (context for your question: paragraph, sentence,etc..)!",
        timeout=60).send()

    if context_res:
        await cl.Message(
            content=f'Your context is:\n{context_res["content"]}.'
        ).send()
    user_context = context_res["content"]

    question_res = await cl.AskUserMessage(content="Now provide your question.", timeout=60,
                                           raise_on_timeout=True).send()

    if question_res:

        await cl.Message(
            content=f'Your question is:\n{question_res["content"]}.'
        ).send()
    user_question = question_res["content"]

    # Your custom logic go here
    msg = cl.Message(content="Please wait for me to process your request!")
    await msg.send()
    answer = make_prediction(context=str(user_context), question=str(user_question))
    await cl.Message(content=answer).send()


@cl.on_chat_end
async def end():
    msg = cl.Message(content="Goodbye. Thank you for visiting me!\nReload the page to start a new session.")
    await msg.send()
