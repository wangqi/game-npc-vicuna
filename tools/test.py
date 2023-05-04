import argparse
import sys

import torch
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer


def main(args):
    parser = argparse.ArgumentParser(args)
    parser.add_argument("--base_model", help="The model to load", default="models/game_npc_vicuna_base")
    parser.add_argument("--lora_model", help="The LoRA weight to load", default="ziqingyang/chinese-alpaca-plus-lora-7b")
    parser.add_argument("--use_lora", action="store_true", help="Whether to use LoRA", default=False)
    parser.add_argument("--model_token", help="The LoRA tokenizer to load", default="config")
    parser.add_argument("--input_text", help="The input text", default="你是一个资深导游，你能介绍一下中国的首都吗?")
    parser.add_argument("--force_cpu", action="store_true", help="Whether to force to use CPU", default=False)
    args = parser.parse_args()
    print(args)

    load_type = torch.float16
    if args.force_cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    print("device:", device)

    base_model = LlamaForCausalLM.from_pretrained(args.base_model,
                                                 load_in_8bit=False,
                                                 torch_dtype=load_type,
                                                 low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_token)
    tokenizer_vocab_size = len(tokenizer)
    print("tokenizer_vocab_size:", tokenizer_vocab_size)
    model_vocab_size = base_model.get_input_embeddings().weight.size(0)
    print("model_vocab_size:", model_vocab_size)
    if model_vocab_size != tokenizer_vocab_size:
        print("The model and tokenizer vocab size do not match!")
        assert tokenizer_vocab_size > model_vocab_size
        base_model.resize_token_embeddings(tokenizer_vocab_size)
        print("Resize the base_model token embedding to:", tokenizer_vocab_size)

    if args.use_lora:
        print("Load the lora weight:", args.lora_model)
        model = PeftModel.from_pretrained(base_model, args.lora_model, torch_type=load_type)
    else:
        print("Only use base model:", args.base_model)
        model = base_model

    model.to(device)
    model.eval()

    generation_config = dict(
        temperature=0.9,
        top_k=40,
        top_p=0.9,
        do_sample=False,
        num_beams=4,
        repetition_penalty=4.0,
        max_new_tokens=400
    )

    input_text = "你是一个资深导游，你能介绍一下中国的首都吗"
    input_ids = tokenizer(input_text, return_tensors="pt")
    generation_output = model.generate(inputs=input_ids["input_ids"].to(device),
                                       attention_mask=input_ids["attention_mask"].to(device),
                                       eos_token_id=tokenizer.eos_token_id,
                                       pad_token_id=tokenizer.pad_token_id,
                                       **generation_config
                                       )
    s = generation_output[0]
    output = tokenizer.decode(s, skip_special_tokens=True)
    print("Input:", input_text)
    print("Output:", output)



if __name__ == '__main__':
    main(sys.argv)
