import os
import sys

import torch
import torch as torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
import argparse
import warnings

assert (
        "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install transformers==4.28.1"
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action="store_true", default=False)
parser.add_argument("--prompt_path", type=str, default="data/prompt_tpl.txt")
parser.add_argument("--data_path", type=str, default="data/data.json")
parser.add_argument("--output_path", type=str, default="models/game_npc_vicuna")
parser.add_argument("--tokenizer_path", type=str, default="config/chinese_llama_alpaca")
parser.add_argument("--model_path", type=str, default="models/game_npc_vicuna_base")
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--eval_steps", type=int, default=200)
parser.add_argument("--save_steps", type=int, default=200)
parser.add_argument("--test_size", type=float, default=0.3)
parser.add_argument("--resume_from_checkpoint", type=str, default=None)
parser.add_argument("--target_models", type=str, default="q_proj,k_proj,v_proj")
parser.add_argument("--micro_batch_size", type=int, default=8)
parser.add_argument("--batch_size", type=int, default=64)
args = parser.parse_args()

if not args.wandb:
    os.environ["WANDB_MODE"] = "disable"

# optimized for RTX 3090 & 4090. for larger GPUs, increase some of these?
MICRO_BATCH_SIZE = args.micro_batch_size
BATCH_SIZE = args.batch_size
MAX_STEPS = None
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = args.num_epochs
#  learning rates of 2e-4, 1e-4, and 5e-5 for the 7B, 13B and 30B models, respectively
LEARNING_RATE = 1e-4
CUTOFF_LEN = 2048 # 13B need about 18G(the cutoff_len can be set to 2048 in 3090Ti/4090Ti)
# LORA_R is the rank of the low-rank adaptation in the LoRA technique.
# It determines the size of the low-rank matrix used in the adaptation.
# A lower rank means a smaller matrix, which requires fewer parameters
# and less computational resources, but may not capture as much information.
# Conversely, a higher rank means a larger matrix, which can capture more
# information but requires more parameters and computational resources.
LORA_R = 8
# LORA_ALPHA is the scaling factor for the low-rank adaptation in the LoRA technique.
# It determines how much the low-rank adaptation contributes to the final output of the model.
# A higher alpha means the low-rank adaptation contributes more, while a lower alpha means it contributes less.
LORA_ALPHA = 16
# LORA_DROPOUT is the dropout rate used in the low-rank adaptation in the LoRA technique.
# Dropout is a regularization technique that randomly sets a fraction of the input units to 0 during training,
# which can help prevent overfitting. The LORA_DROPOUT parameter determines the fraction of
# the input units that are set to 0.
LORA_DROPOUT = 0.05
TEST_SET_SIZE = args.test_size  # 30%
TARGET_MODULES = args.target_models.split(",")
DATA_PATH = args.data_path
OUTPUT_DIR = args.output_path

train_on_inputs: bool = True,  # if False, masks out inputs in loss
add_eos_token: bool = True,
group_by_length: bool = False,  # faster, but produces an odd training loss curve

print(
    f"Training Alpaca-LoRA model with params:\n"
    f"base_model: {args.model_path}\n"
    f"data_path: {DATA_PATH}\n"
    f"output_dir: {OUTPUT_DIR}\n"
    f"batch_size: {BATCH_SIZE}\n"
    f"micro_batch_size: {MICRO_BATCH_SIZE}\n"
    f"num_epochs: {EPOCHS}\n"
    f"learning_rate: {LEARNING_RATE}\n"
    f"cutoff_len: {CUTOFF_LEN}\n"
    f"test_set_ratio: {TEST_SET_SIZE}\n"
    f"lora_r: {LORA_R}\n"
    f"lora_alpha: {LORA_ALPHA}\n"
    f"lora_dropout: {LORA_DROPOUT}\n"
    f"lora_target_modules: {TARGET_MODULES}\n"
    f"train_on_inputs: {train_on_inputs}\n"
    f"add_eos_token: {add_eos_token}\n"
    f"resume_from_checkpoint: {args.resume_from_checkpoint or False}\n"
)

device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
# ddp stands for deep learning process
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size

tokenizer_path = args.tokenizer_path
if not os.path.exists(args.tokenizer_path + "/tokenizer_config.json"):
    tokenizer_path = args.model_path
print("load tokenizer from path:", tokenizer_path)
tokenizer = LlamaTokenizer.from_pretrained(
    tokenizer_path, add_eos_token=add_eos_token
)

print("load model from path:", args.model_path)
max_memory=f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'
print("GPU max memory:", max_memory)
model = LlamaForCausalLM.from_pretrained(
    args.model_path,
    load_in_8bit=True,
    device_map=device_map,
    max_memory=max_memory
)

model = prepare_model_for_int8_training(model)

config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
config.save_pretrained(OUTPUT_DIR)

model = get_peft_model(model, config)
if hasattr(model, 'cpp'):
    # Prevent OOM. https://github.com/Facico/Chinese-Vicuna/issues/81
    model.cpp()
else:
    print("WARNING:", type(model), "does not have cpp() method")
    # sys.exit(-1)

# unk. we want this to be different from the eos token, config
tokenizer.pad_token_id = 0
# tokenizer.padding_side = "left"  # Allow batched inference

total_params, params = 0, 0
for n, p in model.model.named_parameters():
    if any([x in n for x in ["lora"]]):
        total_params += p.numel()
    params += p.numel()

print(
    "Total number of parameters: {}M, rate: {}%".format(
        total_params // 1000 / 1000, round(total_params / params * 100, 2)
    )
)

# Load data
data = load_dataset("json", data_files=DATA_PATH)

TEST_SET_SIZE = int(len(data["train"]) * TEST_SET_SIZE)
VALUE_SET_SIZE = len(data["train"])  - TEST_SET_SIZE
now_max_steps = max(VALUE_SET_SIZE // BATCH_SIZE * EPOCHS, EPOCHS)
print("TEST_SET_SIZE:", TEST_SET_SIZE, "VALUE_SET_SIZE:", VALUE_SET_SIZE, "now_max_steps:", now_max_steps)

if args.resume_from_checkpoint:
    # Check the available weights and load them
    checkpoint_name = os.path.join(args.resume_from_checkpoint, "pytorch_model.bin")  # Full checkpoint
    if not os.path.exists(checkpoint_name):
        pytorch_bin_path = checkpoint_name
        # only LoRA model - LoRA config above has to fit
        checkpoint_name = os.path.join(args.resume_from_checkpoint, "adapter_model.bin")
        if os.path.exists(checkpoint_name):
            os.rename(checkpoint_name, pytorch_bin_path)
            warnings.warn(
                "The file name of the lora checkpoint's adapter_model.bin' is replaced with 'pytorch_model.bin'")
        else:
            args.resume_from_checkpoint = (
                None  # So the trainer won't try loading its state
            )
    # The two files above have a different name depending on how they were saved, but are actually the same.
    if os.path.exists(checkpoint_name):
        print(f"Restarting from {checkpoint_name}")
        adapters_weights = torch.load(checkpoint_name)
        model = set_peft_model_state_dict(model, adapters_weights)
    else:
        print(f"Checkpoint {checkpoint_name} not found")

    train_args_path = os.path.join(args.resume_from_checkpoint, "trainer_state.json")

    if os.path.exists(train_args_path):
        import json

        base_train_args = json.load(open(train_args_path, 'r'))
        base_max_steps = base_train_args["max_steps"]
        resume_scale = base_max_steps / now_max_steps
        if base_max_steps > now_max_steps:
            warnings.warn("epoch {} replace to the base_max_steps {}".format(EPOCHS, base_max_steps))
            EPOCHS = None
            MAX_STEPS = base_max_steps
        else:
            MAX_STEPS = now_max_steps
else:
    MAX_STEPS = now_max_steps

model.print_trainable_parameters()

# Load prompt template
PROMPT_TEMPLATE = ""


def generate_prompt(data_point):
    global PROMPT_TEMPLATE
    if len(PROMPT_TEMPLATE) == 0:
        # use os to check if file exists
        if os.path.exists(args.prompt_path):
            with open(args.prompt_path, 'r') as f:
                PROMPT_TEMPLATE = f.read()
        else:
            raise FileNotFoundError(args.prompt_path)
    user_prompt = PROMPT_TEMPLATE.format(instruction=data_point["instruction"], input=data_point["input"],
                                         output=data_point["output"])
    return user_prompt


def tokenize(prompt):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )
    return {
        "input_ids": result["input_ids"][:-1],
        "attention_mask": result["attention_mask"][:-1],
    }


def generate_and_tokenize_prompt(data_point):
    # This function masks out the labels for the input,
    # so that our loss is computed only on the response.
    user_prompt = generate_prompt(data_point)
    # print("user_prompt: ", user_prompt)
    len_user_prompt_tokens = (
            len(
                tokenizer(
                    user_prompt,
                    truncation=True,
                    max_length=CUTOFF_LEN + 1,
                )["input_ids"]
            )
            - 1
    )  # no eos token
    full_tokens = tokenizer(
        user_prompt + data_point["output"],
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )["input_ids"][:-1]
    return {
        "input_ids": full_tokens,
        "labels": [-100] * len_user_prompt_tokens
                  + full_tokens[len_user_prompt_tokens:],
        "attention_mask": [1] * (len(full_tokens)),
    }

if TEST_SET_SIZE > 0:
    train_val = data["train"].train_test_split(
        test_size=TEST_SET_SIZE, shuffle=True, seed=42
    )
    train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
else:
    train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
    val_data = None

print("Training dataset size:", len(train_data), ". Validation dataset size:", len(val_data))
trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        per_device_eval_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=20,
        evaluation_strategy="steps" if TEST_SET_SIZE > 0 else "no",
        save_strategy="steps",
        eval_steps=args.eval_steps if TEST_SET_SIZE > 0 else None,
        save_steps=args.save_steps,
        output_dir=OUTPUT_DIR,
        save_total_limit=3,
        load_best_model_at_end=True if TEST_SET_SIZE > 0 else False,
        ddp_find_unused_parameters=False if ddp else None,
        report_to="wandb" if args.wandb else [],
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
).__get__(model, type(model))

if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

print("\n If there's a warning about missing keys above, please disregard :)")

if args.resume_from_checkpoint:
    print("resume from checkpoint")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
else:
    trainer.train()

model.save_pretrained(OUTPUT_DIR)
