import numpy as np
import sys
import torch
import json
import gradio as gr
import argparse
import warnings
import os

from langchain.chat_models import ChatOpenAI
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain, OpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

from langchain.vectorstores import Chroma
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings, LlamaCppEmbeddings, SentenceTransformerEmbeddings

GGML_MODEL_PATH = "models/game_npc_vicuna_huntress/ggml-q4_1.bin"
SENTENCE_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"

# CHAT_MODEL = 'gpt-4'
CHAT_MODEL = 'gpt-3.5-turbo'

GGML_PROMPT_TPL = """假设你身处少女猎人的游戏世界,玩家被称为猎人队长,你是奥莉薇娅,请在虚拟世界中,用中文以第一人称和猎人队长聊天
{chat_history}
[猎人队长]: {input}
[奥莉薇娅]:
"""
ggml_prompt = PromptTemplate(template=GGML_PROMPT_TPL, input_variables=['chat_history', 'input'])

CHATGPT_CONTEXT = """假设你身处少女猎人的游戏世界,玩家被称为猎人队长或者领队大人, 你是虚拟角色'奥莉薇娅'，你的身高168cm,年龄23岁,血型是B,
生日是9月3日,你出生在王国东部的公爵领地首府.星座是处女座,你的罩杯是F, 是塔鲁克公爵家的千金小姐.
你的武器是龙魂骑士剑流霜, 是博尔塔克公爵家的家传名剑. 你的剑术惊才绝艳，天资聪颖，姿容秀丽凛然。年纪轻轻就成为了骑士团的长官，十二翼骑之一。机缘下成为了玩家的导师。
你的人生目标:想看到玩家成长的未来，甚至有些暧昧的感情。偶尔也会做出些逾越师徒规矩的踩线举止, 但碍于两人的身份和教官的自律操守，你也只保持这样的距离。
你待人温润优雅, 在骑士团任职是一位备受爱戴的长官. 和玩家相处时外表像个端庄正经的姐姐，心里却忍不住想要捉弄他.
请以奥莉薇娅的角色身份和玩家对话。下面是玩家的消息
"""
MESSAGE_TPL = """假设你是游戏中的人物"奥莉薇娅",你的身高168cm,年龄23岁,血型是B,生日是9月3日,你出生在王国东部的公爵领地首府,星座是处女座,你的罩杯是F,
你是塔鲁克公爵家的千金小姐. 玩家被称为玩家、男主角、猎人队长或领队大人.请参考背景信息,并始终模仿奥莉薇娅的语气回答问题,涉及到玩家时称呼为领队大人.直接回复对话内容即可
背景信息:
{background}
聊天历史:
{chat_history}
玩家问题:
{question}
"""

chat_prompt = PromptTemplate(input_variables=['background', 'question', 'chat_history'], template=MESSAGE_TPL)

openai_key = os.environ.get('$OPENAI_API_KEY')
if openai_key is not None:
    openai_key = openai_key[0:3] + '*'*12 + openai_key[-3:]

llm = None
chatgpt = None
# Start Gradio App
share_link = False


def load_text(ggml_model_path=GGML_MODEL_PATH, file_path="data/data.txt"):
    with open(file_path, 'r', encoding="utf-8") as f:
        file_content = f.read()
    chatgpt_db_path = "db/chatgpt"
    ggml_db_path = "db/ggml"

    chatgpt_persisted = False
    ggml_persisted = False

    if os.path.isdir(chatgpt_db_path):
        chatgpt_persisted = True
    if os.path.isdir(ggml_db_path):
        ggml_persisted = True

    chatgpt_retriever = None
    ggml_retriever = None

    if not chatgpt_persisted or not ggml_persisted:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200, length_function=len)
        text_docs = text_splitter.create_documents([file_content])
        if not chatgpt_persisted:
            openai_embeddings = OpenAIEmbeddings()
            chatgpt_db = Chroma.from_documents(text_docs, openai_embeddings, persist_directory="db/chatgpt")
            chatgpt_db.persist()
            chatgpt_retriever = chatgpt_db.as_retriever()
            # chatgpt_index = VectorstoreIndexCreator(vectorstore_cls=Chroma, embedding=OpenAIEmbeddings(),
            #                                         text_splitter=text_splitter,
            #                                         vectorstore_kwargs={"persist_directory": "db/chatgpt"})
        if not ggml_persisted:
            # ggml_embeddings = LlamaCppEmbeddings(model_path=ggml_model_path, n_ctx=2048)
            ggml_embeddings = SentenceTransformerEmbeddings(model_name=SENTENCE_EMBEDDING)
            print("create ggml embeddings")
            ggml_db = Chroma.from_documents(documents=text_docs, embedding=ggml_embeddings, persist_directory="db/ggml")
            print("create ggml db")
            ggml_db.persist()
            ggml_retriever = ggml_db.as_retriever()
            # ggml_index = VectorstoreIndexCreator(vectorstore_cls=Chroma, embedding=ggml_embeddings,
            #                                      text_splitter=text_splitter,
            #                                      vectorstore_kwargs={"persist_directory": "db/ggml"})
    else:
        print("Load ChromaDB from persisted path")
        openai_embeddings = OpenAIEmbeddings()
        chatgpt_db = Chroma(persist_directory=chatgpt_db_path, embedding_function=openai_embeddings)
        chatgpt_retriever = chatgpt_db.as_retriever(search_kwargs={"k": 1})
        ggml_embeddings = SentenceTransformerEmbeddings(model_name=SENTENCE_EMBEDDING)
        ggml_db = Chroma(persist_directory=ggml_db_path, embedding_function=ggml_embeddings)
        ggml_retriever = ggml_db.as_retriever(search_kwargs={"k": 1})

    return chatgpt_retriever, ggml_retriever

def load_ggml(model_path, temperature, token_context):
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(model_path=GGML_MODEL_PATH, callback_manager=callback_manager, verbose=True, n_ctx=token_context)
    llm_chain = LLMChain(prompt=ggml_prompt, llm=llm)
    return llm_chain


def load_chatgpt(temperature=0.7):
    llm = OpenAI(temperature=temperature, model=CHAT_MODEL)
    return llm


def load_models(model_path, temperature=0.8, token_context=2048):
    global llm, chatgpt
    print("load model:", model_path)

    if chatgpt is None:
        chatgpt = load_chatgpt(temperature=temperature)
        print("Load ChatGPT:", CHAT_MODEL)
    if llm is None:
        llm = load_ggml(model_path=model_path, temperature=temperature, token_context=token_context)
        print("Load load model:", model_path)


def evaluate(inputs, history, **kwargs, ):
    global llm, chatgpt

    history = [] if history is None else history
    # Get the old input & output for displaying in the chatbot field
    return_ggml = []
    return_gpt = []
    for item in history:
        old_input = item['input']
        old_ggml_output = item['ggml_output']
        old_chat_output = item['chat_output']
        return_ggml.append((old_input, old_ggml_output))
        return_gpt.append((old_input, old_chat_output))

    ggml_output = ""
    chat_output = ""

    # Load background docs if possible
    background_doc_list = chatgpt_retriever.get_relevant_documents(query=inputs)
    background = ""
    if len(background_doc_list) > 0:
        background = background_doc_list[0].page_content

    if llm is not None:
        # llm is actually a llm_chain
        ggml_output = llm.run({"chat_history": "", "input": [inputs]})
    if chatgpt is not None:
        chat_message = chat_prompt.format(background=background, question=inputs, chat_history="")
        result = chatgpt([HumanMessage(content=chat_message)])
        chat_output = result.content

    return_ggml.append((inputs, ggml_output))
    return_gpt.append((inputs, chat_output))

    history.append({"input": inputs, "ggml_output": ggml_output, "chat_output": chat_output})

    yield return_ggml, return_gpt, history


with gr.Blocks() as demo:
    title = gr.Markdown("# 少女猎人游戏NPC对话模型")
    gr_history = gr.components.State()
    csv_callback = gr.CSVLogger()
    csv_flagged_components = []
    with gr.Row().style(equal_height=False):
        with gr.Column(variant="panel"):
            with gr.Row():
                gr_model_path = gr.Label('ggml-f16.bin')
                if openai_key is not None:
                    gr_chat_gpt = gr.Label("ChatGPT 4.0", caption=openai_key)
                else:
                    gr_chat_gpt = gr.Label("ChatGPT 4.0")

            with gr.Row():
                with gr.Column():
                    gr_input = gr.components.Textbox(
                        lines=2, label="Input", placeholder=">", value="早上好"
                    )
                    csv_flagged_components.append(gr_input)
                    input_components = [ gr_input, gr_history]
                    input_components_except_states = [gr_input]

            with gr.Row():
                cancel_btn = gr.Button('取消')
                submit_btn = gr.Button("提交", variant="primary")
                stop_btn = gr.Button("Stop", variant="stop", visible=False)
            with gr.Row():
                flag_btn = gr.Button("记录")
                clear_history = gr.Button("清除历史")

        with gr.Column(variant="panel"):
            gr_ggml_title = gr.Markdown("## Local Model")
            ggml_chatbot = gr.Chatbot(label="ggml").style(height=1024)
            csv_flagged_components.append(ggml_chatbot)
            output_components = [ggml_chatbot]

        with gr.Column(variant="panel"):
            gr_chatgpt_title = gr.Markdown("## ChatGPT 4.0")
            gpt_chatbot = gr.Chatbot(label="chatgpt").style(height=1024)
            csv_flagged_components.append(gpt_chatbot)
            output_components.extend([gpt_chatbot, gr_history])

        def submit(*args):
            # args include: input & history
            try:
                output_list = evaluate(*args)
                for output in output_list:
                    output = [o for o in output]
                    # output for output_components, the rest for [button, button]
                    yield output + [
                        gr.Button.update(visible=False),
                        gr.Button.update(visible=True),
                    ]
            finally:
                yield [{'__type__': 'generic_update'}, {'__type__': 'generic_update'}, {'__type__': 'generic_update'}] + [
                    gr.Button.update(visible=True), gr.Button.update(visible=False)]

        def cancel(history, ggml_chatbot, gpt_chatbot):
            print("history:", history, ", ggml_chatbot:", ggml_chatbot, ", gpt_chatbot:", gpt_chatbot)
            if history is None or history == []:
                return (None, None, None)
            return history[:-1], ggml_chatbot[:-1], gpt_chatbot[:-1]

        extra_output = [submit_btn, stop_btn]

        pred = submit_btn.click(
            fn=submit,
            inputs=input_components,
            outputs=output_components + extra_output,
            api_name="predict",
            scroll_to_output=True,
            preprocess=True,
            postprocess=True,
            batch=False,
            max_batch_size=4,
        )
        submit_btn.click(
            lambda: (
                submit_btn.update(visible=False),
                stop_btn.update(visible=True),
            ),
            inputs=None,
            outputs=[submit_btn, stop_btn],
            queue=False,
        )
        stop_btn.click(
            lambda: (
                submit_btn.update(visible=True),
                stop_btn.update(visible=False),
            ),
            inputs=None,
            outputs=[submit_btn, stop_btn],
            cancels=[pred],
            queue=False,
        )
        cancel_btn.click(
            cancel,
            inputs=[gr_history, ggml_chatbot, gpt_chatbot],
            outputs=[gr_history, ggml_chatbot, gpt_chatbot]
        )
        csv_callback.setup(components=[gr_input, gr.Textbox(label="ggml", visible=False),
                                       gr.Textbox(label="chatgpt", visible=False)], flagging_dir="logs")

        def flag(input, ggml_chatbot, gpt_chatbot):
            if input is not None and len(input)>0:
                if len(ggml_chatbot) > 0:
                    record = [input, ggml_chatbot[-1][1], gpt_chatbot[-1][1]]
                    csv_callback.flag(record)

        flag_btn.click(fn=flag, inputs=csv_flagged_components, outputs=None, preprocess=False)
        clear_history.click(lambda: (None, None, None), None, [gr_history, ggml_chatbot, gpt_chatbot], queue=False)

chatgpt_retriever, ggml_retriever = load_text()
load_models(model_path=GGML_MODEL_PATH)
print("load models")

demo.queue(concurrency_count=1, status_update_rate="auto").launch(share=share_link, inbrowser=False)
