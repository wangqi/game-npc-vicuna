"""
Borrowed and modified from https://github.com/tloen/alpaca-lora
"""

import argparse

import torch
import transformers
import peft
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--base_model', default="decapoda-research/llama-7b-hf", required=True,
                    type=str, help="Please specify a base_model")
parser.add_argument('--lora_model', default="lora_out/final", required=True,
                    type=str, help="Please specify a lora_model")
parser.add_argument('--lora_token', default="config", required=True,
                    type=str, help="Please specify a lora_model")
parser.add_argument('--offload_dir', default="/mnt/c/offload_dir", type=str,
                    help="(Optional) Please specify a temp folder for offloading (useful for low-RAM machines). Default None (disable offload).")
parser.add_argument('--output_dir', default='./models/', type=str)
args = parser.parse_args()


assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"

BASE_MODEL = args.base_model
LORA_MODEL = args.lora_model
LOAD_8BIT = False
OUTPUT_DIR = args.output_dir

assert (
    BASE_MODEL
), "Please specify a BASE_MODEL in the script, e.g. 'huggyllama/llama-7b'"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("device: ", device)

tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
if device == "cuda":
    base_model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=LOAD_8BIT,
        torch_dtype=torch.float16,
        device_map={"": 0},
    )
else:
    if args.offload_dir is not None:
        # Load with offloading, which is useful for low-RAM machines.
        # Note that if you have enough RAM, please use original method instead, as it is faster.
        base_model = LlamaForCausalLM.from_pretrained(
            BASE_MODEL,
            load_in_8bit=LOAD_8BIT,
            torch_dtype=torch.float16,
            offload_folder=args.offload_dir,
            offload_state_dict=True,
            low_cpu_mem_usage=True,
            device_map={"": device},
        )
    else:
        # Original method without offloading
        base_model = LlamaForCausalLM.from_pretrained(
            BASE_MODEL,
            load_in_8bit=LOAD_8BIT,
            torch_dtype=torch.float16,
            device_map={"": device},
        )

base_model.resize_token_embeddings(len(tokenizer))
assert base_model.get_input_embeddings().weight.size(0) == len(tokenizer)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Extended vocabulary size: {len(tokenizer)}")

first_weight = base_model.model.layers[0].self_attn.q_proj.weight
first_weight_old = first_weight.clone()

if device == 'cuda':
    lora_model = PeftModel.from_pretrained(
        base_model,
        LORA_MODEL,
        device_map={"": 0},
        torch_dtype=torch.float16,
    )
else:
    lora_model = PeftModel.from_pretrained(
        base_model,
        LORA_MODEL,
        device_map={"": "cpu"},
        torch_dtype=torch.float16,
    )

assert torch.allclose(first_weight_old, first_weight)

print(f"Peft version: {peft.__version__}")
print(f"Merging model")
if peft.__version__ > '0.2.0':
    # merge weights - new merging method from peft
    lora_model = lora_model.merge_and_unload()
else:
    # merge weights
    for layer in lora_model.base_model.model.model.layers:
        if hasattr(layer.self_attn.q_proj,'merge_weights'):
            layer.self_attn.q_proj.merge_weights = True
        if hasattr(layer.self_attn.v_proj,'merge_weights'):
            layer.self_attn.v_proj.merge_weights = True
        if hasattr(layer.self_attn.k_proj,'merge_weights'):
            layer.self_attn.k_proj.merge_weights = True
        if hasattr(layer.self_attn.o_proj,'merge_weights'):
            layer.self_attn.o_proj.merge_weights = True
        if hasattr(layer.mlp.gate_proj,'merge_weights'):
            layer.mlp.gate_proj.merge_weights = True
        if hasattr(layer.mlp.down_proj,'merge_weights'):
            layer.mlp.down_proj.merge_weights = True
        if hasattr(layer.mlp.up_proj,'merge_weights'):
            layer.mlp.up_proj.merge_weights = True

lora_model.train(False)

# did we do anything?
assert not torch.allclose(first_weight_old, first_weight)

lora_model_sd = lora_model.state_dict()
deloreanized_sd = {
    k.replace("base_model.model.", ""): v
    for k, v in lora_model_sd.items()
    if "lora" not in k
}
LlamaForCausalLM.save_pretrained(
    base_model, OUTPUT_DIR, state_dict=deloreanized_sd
)
