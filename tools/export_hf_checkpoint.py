"""
Copy from https://github.com/tloen/alpaca-lora/blob/main/export_hf_checkpoint.py
"""
import argparse
import os

import torch
import transformers
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer  # noqa: F402

parser = argparse.ArgumentParser()
parser.add_argument("--base_model", type=str, default="decapoda-research/llama-8b-hf")
parser.add_argument("--lora_model", type=str, default="tloen/alpaca-lora-7b")
parser.add_argument("--data_path", type=str, default="data/data.json")
parser.add_argument("--output_path", type=str, default="models/game_npc_vicuna")
parser.add_argument("--model_path", type=str, default="models/game_npc_vicuna_base")
args = parser.parse_args()

BASE_MODEL = args.base_model
LORA_MODEL = args.lora_model
print("base_model:", BASE_MODEL)

tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
base_model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map="auto",
)

first_weight = base_model.model.layers[0].self_attn.q_proj.weight
first_weight_old = first_weight.clone()

lora_model = PeftModel.from_pretrained(
    base_model,
    LORA_MODEL,
    device_map="auto",
    torch_dtype=torch.float16,
)

lora_weight = lora_model.base_model.model.model.layers[0].self_attn.q_proj.weight

assert torch.allclose(first_weight_old, first_weight)

# merge weights - new merging method from peft
lora_model = lora_model.merge_and_unload()
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
    base_model, "./models/hf_ckpt", state_dict=deloreanized_sd, max_shard_size="4000MB"
)
