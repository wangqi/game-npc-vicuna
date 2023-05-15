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
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory


PROMPT_TPL = """假设你身处少女猎人的游戏世界,玩家被称为猎人队长或者领队大人, 根据Instruction中的描述，用中文以第一人称回答
{chat_history}
### Instruction:{input}
### Response:
"""

llm_chain = None
# Start Gradio App
share_link = False


def load_ggml():
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(model_path="models/game_npc_vicuna_huntress/ggml-f16.bin",
                   callback_manager=callback_manager, verbose=True)
    return llm

# Load the ggml model in advance
ggml_llm = load_ggml()

def load_model(model_path="models/game_npc_vicuna_huntress/ggml-f16.bin", temperature=0.8, token_context=2000):
    global llm_chain
    print("load model:", model_path)

    if model_path == 'ChatGPT':
        llm = ChatOpenAI(temperature=temperature)
    else:
        global ggml_llm
        if ggml_llm is None:
            ggml_llm = load_ggml()
            print("load model:", model_path)
        llm = ggml_llm

    prompt = PromptTemplate(template=PROMPT_TPL, input_variables=['chat_history', 'input'])
    # memory = ConversationBufferMemory(memory_key="chat_history")

    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)
    return llm_chain


def evaluate(inputs, history, **kwargs, ):
    global llm_chain

    history = [] if history is None else history
    return_text = [(item['input'], item['output']) for item in history]
    output = llm_chain.run({"chat_history": "", "input": [inputs]})
    return_text += [(inputs, output)]
    history.append({"input": inputs, "output": output})

    yield return_text, history


with gr.Blocks() as demo:
    title = gr.Markdown("# 少女猎人游戏NPC对话模型")
    gr_history = gr.components.State()
    with gr.Row().style(equal_height=False):
        with gr.Column(variant="panel"):
            with gr.Row():
                gr_model_path = gr.Dropdown(choices=['ggml-f16.bin', 'ChatGPT'], value='ggml-f16.bin', label="选择模型")
            with gr.Row():
                gr_temperature = gr.components.Slider(minimum=0, maximum=1, value=0.6, label="Temperature")
            with gr.Row():
                gr_load_model_btn = gr.Button("加载模型", variant="secondary")

            with gr.Row():
                input_component_column = gr.Column()
                with input_component_column:
                    gr_input = gr.components.Textbox(
                        lines=2, label="Input", placeholder=">"
                    )
                    input_components = [ gr_input, gr_history]
                    input_components_except_states = [gr_input]

            with gr.Row():
                cancel_btn = gr.Button('Cancel')
                submit_btn = gr.Button("Submit", variant="primary")
                stop_btn = gr.Button("Stop", variant="stop", visible=False)
            with gr.Row():
                reset_btn = gr.Button("Reset Parameter")
                clear_history = gr.Button("Clear History")

        with gr.Column(variant="panel"):
            chatbot = gr.Chatbot().style(height=1024)
            output_components = [chatbot, gr_history]

        def submit(*args):
            # here to support the change between the stop and submit button
            try:
                output_list = evaluate(*args)
                for output in output_list:
                    print(output)
                    output = [o for o in output]
                    # output for output_components, the rest for [button, button]
                    yield output + [
                        gr.Button.update(visible=False),
                        gr.Button.update(visible=True),
                    ]
            finally:
                yield [{'__type__': 'generic_update'}, {'__type__': 'generic_update'}] + [
                    gr.Button.update(visible=True), gr.Button.update(visible=False)]

        def cancel(history, chatbot):
            if history == []:
                return (None, None)
            return history[:-1], chatbot[:-1]

        extra_output = [submit_btn, stop_btn]

        gr_load_model_btn.click(
            fn=load_model,
            inputs=[gr_model_path, gr_temperature],
            outputs=None,
            api_name="load_model",
            postprocess=False
        )

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
            inputs=[gr_history, chatbot],
            outputs=[gr_history, chatbot]
        )
        reset_btn.click(fn=None, inputs=[], outputs=(
            # input_components ; don't work for history...
            input_components_except_states + [input_component_column]),
            # type: ignore
            _js=f"""() => {json.dumps([
                                getattr(component, "cleared_value", None) for component in input_components_except_states]
                                      + ([gr.Column.update(visible=True)])
                                      + ([])
                                      )}
            """,
        )
        clear_history.click(lambda: (None, None), None, [gr_history, chatbot], queue=False)

demo.queue(concurrency_count=1, status_update_rate="auto").launch(share=share_link, inbrowser=False)
