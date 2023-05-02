#BASE_MODEL="decapoda-research/llama-7b-hf" #"/model/13B_hf"
BASE_MODEL="models/game_npc_vicuna_base"
LORA_PATH="lora_out/final-7b" #"checkpoint-6000"
USE_LOCAL=1 # 1: use local model, 0: use huggingface model
DEBUG=0
if [[ USE_LOCAL -eq 1 ]]
then
  echo "Use local LORA: $LORA_PATH"
else
  echo "Use Remote LORA in Huggingface: $LORA_PATH"
fi

if [[ DEBUG -eq 1 ]]
then
    jurigged -v tools/chat.py --model_path $BASE_MODEL --lora_path $LORA_PATH --use_local $USE_LOCAL\
    --use_typewriter 0 \
    --share_link 0 --load_lora
else
CUDA_VISIBLE_DEVICES=0 python tools/chat.py --model_path $BASE_MODEL --lora_path $LORA_PATH --use_local $USE_LOCAL\
    --use_typewriter 0 \
    --share_link 0 --load_lora
fi