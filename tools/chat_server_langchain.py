import numpy as np
import sys
import torch
import json
import gradio as gr
import argparse
import warnings
import os, textwrap

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
from langchain import HuggingFacePipeline

GGML_MODEL_PATH = "models/game_npc_vicuna_huntress/ggml-q4_1.bin"
# HF_MODEL_PATH = "models/wizard-mega_chinese-alpaca-plus-lora-13b"
HF_MODEL_PATH = "models/yahma_chinese-alpaca-plus-lora-13b"
SENTENCE_EMBEDDING = "shibing624/text2vec-base-chinese"

CHAT_MODEL = 'gpt-4'
# CHAT_MODEL = 'gpt-3.5-turbo'
FASTCHAT_MODEL = 'huntress'
FASTCHAT_API_BASE="http://localhost:8000/v1"


CHATGPT_TPL = """假设你身处少女猎人的游戏世界,玩家被称为猎人队长,你是奥莉薇娅,请用中文以第一人称回复猎人队长的问题,回答中不要跳出设定。
背景信息:
{background}
聊天历史:
{chat_history}
[猎人队长]: {question}
"""

MESSAGE_TPL = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
假设你身处少女猎人的游戏世界,玩家被称为猎人队长,你是奥莉薇娅,请用中文以第一人称回复猎人队长的问题,回答中不要跳出设定。
背景信息:
{background}
聊天历史:
{chat_history}
[猎人队长]: {question}
### Response:
"""

# chat_prompt = PromptTemplate(input_variables=['background', 'question', 'chat_history'], template=MESSAGE_TPL)
# ggml_prompt = PromptTemplate(input_variables=['background', 'question', 'chat_history'], template=MESSAGE_TPL)

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
        # Convert to absolute path
        chatgpt_db_path = os.path.abspath(chatgpt_db_path)
        print("found chatgpt db at path:", chatgpt_db_path)
        chatgpt_persisted = True
    if os.path.isdir(ggml_db_path):
        # Convert to absolute path
        ggml_db_path = os.path.abspath(ggml_db_path)
        print("found ggml db at path:", ggml_db_path)
        ggml_persisted = True

    chatgpt_retriever = None
    ggml_retriever = None

    if not chatgpt_persisted or not ggml_persisted:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200, length_function=len)
        text_docs = text_splitter.create_documents([file_content])
        print("total docs:", len(text_docs))
        if not chatgpt_persisted:
            openai_embeddings = OpenAIEmbeddings()
            chatgpt_db = Chroma.from_documents(text_docs, openai_embeddings, persist_directory=chatgpt_db_path)
            chatgpt_db.persist()
            chatgpt_retriever = chatgpt_db.as_retriever()
        if not ggml_persisted:
            ggml_embeddings = SentenceTransformerEmbeddings(model_name=SENTENCE_EMBEDDING)
            print("create ggml embeddings")
            ggml_db = Chroma.from_documents(documents=text_docs, embedding=ggml_embeddings, persist_directory=ggml_db_path)
            print("create ggml db")
            ggml_db.persist()
            ggml_retriever = ggml_db.as_retriever(search_type="mmr")

    else:
        print("Load ChromaDB from persisted path")
        openai_embeddings = OpenAIEmbeddings()
        chatgpt_db = Chroma(persist_directory=chatgpt_db_path, embedding_function=openai_embeddings)
        chatgpt_retriever = chatgpt_db.as_retriever()
        print("Load ggml chromadb from ", ggml_db_path, " with embeddings:", SENTENCE_EMBEDDING)
        ggml_embeddings = SentenceTransformerEmbeddings(model_name=SENTENCE_EMBEDDING)
        ggml_db = Chroma(persist_directory=ggml_db_path, embedding_function=ggml_embeddings)
        ggml_retriever = ggml_db.as_retriever(search_type="mmr")

    return chatgpt_retriever, ggml_retriever

def load_ggml(model_path, temperature, token_context):
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(model_path=GGML_MODEL_PATH, callback_manager=callback_manager, verbose=True, n_ctx=token_context)
    # llm_chain = LLMChain(prompt=ggml_prompt, llm=llm)
    # return llm_chain
    return llm

def load_chatgpt(temperature=0.7):
    llm = ChatOpenAI(temperature=temperature, model=CHAT_MODEL)
    return llm

def load_hf(model_path, temperature, token_context):
    llm = HuggingFacePipeline.from_model_id(model_id=model_path, task="text-generation",
                                            model_kwargs={"temperature": temperature,
                                                          "load_in_8bit": True,
                                                          "max_length": 1024,
                                                          "device_map": 'auto'})
    return llm

def load_fastchat(temperature=0.7):
    llm = ChatOpenAI(temperature=temperature, model=FASTCHAT_MODEL, openai_api_base=FASTCHAT_API_BASE)
    return llm


def load_models(model_path, temperature=0.8, token_context=2048):
    global llm, chatgpt
    print("load model:", model_path)

    if chatgpt is None:
        chatgpt = load_chatgpt(temperature=temperature)
        print("Load ChatGPT:", CHAT_MODEL)
    if llm is None:
        # llm = load_ggml(model_path=model_path, temperature=temperature, token_context=token_context)
        llm = load_hf(model_path=model_path, temperature=temperature, token_context=token_context)
        print("Load HuggingFace local pipeline model:", model_path)
        # llm = load_fastchat(temperature=temperature)
        # print("Load FastChat API model:", FASTCHAT_API_BASE)


def query_retriever(query, retriever, use_textwrap=True, max_tokens=200):
    # Load background docs if possible
    background_doc_list = retriever.get_relevant_documents(query=query)
    if len(background_doc_list) > 0:
        background = background_doc_list[0].page_content
        if len(background) > max_tokens:
            background = background[0:max_tokens]
        if use_textwrap:
            return "\n".join(textwrap.wrap(background, width=80))
        return background
    return ""


def evaluate(query, history, prompt, enable_background, enable_history, **kwargs, ):
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

    ggml_history = '\n'.join(f'[猎人队长]{t[0]}\n[奥莉薇娅]{t[1]}' for t in return_ggml)
    ggml_history = ggml_history[:-256]
    chatgpt_history = '\n'.join(f'[猎人队长]{t[0]}\n[奥莉薇娅]{t[1]}' for t in return_gpt)
    chatgpt_history = chatgpt_history[:-512]
    print("ggml_history:", ggml_history)
    print("chatgpt_history:", chatgpt_history)

    ggml_output = ""
    chat_output = ""

    if prompt is None or len(prompt) == 0:
        prompt = "{background}{chat_history}{question}"
    chat_prompt = PromptTemplate(input_variables=['background', 'question', 'chat_history'], template=prompt)

    if llm is not None:
        # llm is actually a llm_chain
        if enable_background:
            background = query_retriever(query=query, retriever=ggml_retriever, use_textwrap=True, max_tokens=512)
        else:
            print("ggml background is disabled")
            background = ""
        print("ggml background:", background)

        ggml_prompt = PromptTemplate(input_variables=['background', 'chat_history', 'question'],
                                     template=MESSAGE_TPL)
        llm_chain = LLMChain(prompt=ggml_prompt, llm=llm)
        if not enable_history:
            ggml_output = llm_chain({"background": background, "chat_history": "", "question": query},
                                    return_only_outputs=True).get('text', '')
        else:
            ggml_output = llm_chain({"background": background, "chat_history": ggml_history, "question": query},
                                    return_only_outputs=True).get('text', '')
        output_lines = ggml_output.split("[猎人队长]:")
        print("ggml output before replacing:", ggml_output)
        for output_line in output_lines:
            if "[奥莉薇娅]" in output_line:
                ggml_output = ggml_output.replace("[奥莉薇娅]", "")
            if ":" in ggml_output:
                ggml_output = ggml_output.replace(":", "")
            break
        print("ggml output after replacing:", ggml_output)
    if chatgpt is not None:
        # Load background docs if possible
        if enable_background:
            background = query_retriever(query=query, retriever=chatgpt_retriever, use_textwrap=True, max_tokens=1024)
        else:
            background = ""
        print("chatgpt background:", background)
        if not enable_history:
            chat_message = chat_prompt.format(background=background, question=query, chat_history="")
        else:
            chat_message = chat_prompt.format(background=background, question=query, chat_history=chatgpt_history)
        result = chatgpt([HumanMessage(content=chat_message)])
        chat_output = result.content
        if "[奥莉薇娅]" in chat_output:
            chat_output = chat_output.replace("[奥莉薇娅]", "")
        if ":" in ggml_output:
            chat_output = chat_output.replace(":", "")
        print("chatgpt output:", chat_output)

    return_ggml.append((query, ggml_output))
    return_gpt.append((query, chat_output))

    history.append({"input": query, "ggml_output": ggml_output, "chat_output": chat_output})

    yield return_ggml, return_gpt, history


with gr.Blocks() as demo:
    title = gr.Markdown("# 少女猎人游戏NPC对话模型")
    gr_history = gr.components.State()
    csv_callback = gr.CSVLogger()
    csv_flagged_components = []
    with gr.Row().style(equal_height=False):
        with gr.Column(variant="panel"):
            with gr.Row():
                gr_model_path = gr.Label("自训练模型(13B)")
                if openai_key is not None:
                    if CHAT_MODEL == "gpt-3.5-turbo":
                        gr_chat_gpt = gr.Label("ChatGPT 3.5", caption=openai_key)
                    else:
                        gr_chat_gpt = gr.Label("ChatGPT 4.0", caption=openai_key)
                else:
                    if CHAT_MODEL == "gpt-3.5-turbo":
                        gr_chat_gpt = gr.Label("ChatGPT 3.5")
                    else:
                        gr_chat_gpt = gr.Label("ChatGPT 4.0")

            with gr.Row():
                gr_prompt_input = gr.components.Textbox(
                    lines=4, label="Prompt", placeholder="", value=MESSAGE_TPL
                )

            with gr.Row():
                gr_enable_background_checkbox = gr.Checkbox(label="启用故事背景", value=True)
                gr_enable_history_checkbox = gr.Checkbox(label="启用聊天历史", value=True)

            with gr.Row():
                with gr.Column():
                    gr_input = gr.components.Textbox(
                        lines=2, label="Input", placeholder=">", value="早上好"
                    )
                    csv_flagged_components.append(gr_input)
                    input_components = [ gr_input, gr_history, gr_prompt_input, gr_enable_background_checkbox, gr_enable_history_checkbox ]
                    input_components_except_states = [gr_input, gr_prompt_input, gr_enable_background_checkbox, gr_enable_history_checkbox]

            with gr.Row():
                cancel_btn = gr.Button('取消')
                submit_btn = gr.Button("提交", variant="primary")
                stop_btn = gr.Button("Stop", variant="stop", visible=False)
            with gr.Row():
                flag_btn = gr.Button("记录")
                clear_history = gr.Button("清除历史")

        with gr.Column(variant="panel"):
            # gr_ggml_title = gr.Markdown("## " + HF_MODEL_PATH.split("/")[-1][0:20])
            gr_ggml_title = gr.Markdown("## " + "130亿参数本地模型")
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
load_models(model_path=HF_MODEL_PATH)
print("load models")

demo.queue(concurrency_count=1, status_update_rate="auto").launch(share=share_link, inbrowser=False)
