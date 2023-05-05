#BASE_MODEL="models/game_npc_vicuna_base"
BASE_MODEL="models/huggyllama_chinese-alpaca-plus-lora-7b"
#BASE_MODEL="huggyllama/llama-7b"
#BASE_MODEL="decapoda-research/llama-7b-hf"
#BASE_MODEL="models/game_npc_vicuna_base"
#BASE_MODEL="models/huggyllama_chinese-vicuna-7b-3epock-belle-guanaco-chinese-llama-alpaca"

#LORA_PATH="ziqingyang/chinese-alpaca-plus-lora-7b"
#LORA_PATH="Chinese-Vicuna/Chinese-Vicuna-lora-7b-belle-and-guanaco"
#LORA_PATH="Facico/Chinese-Vicuna-lora-7b-3epoch-belle-and-guanaco"
LORA_PATH="lora_out/huntress-7b" #"checkpoint-6000"

USE_LOCAL=1 # 1: use local model, 0: use huggingface model
DEBUG=0
if [[ USE_LOCAL -eq 1 ]]
  echo "Use local LORA: $LORA_PATH"
then
  echo "Use Remote LORA in Huggingface: $LORA_PATH"
fi

if [[ DEBUG -eq 1 ]]
then
    jurigged -v tools/chat_server.py --model_path $BASE_MODEL --lora_path $LORA_PATH --use_local $USE_LOCAL\
    --use_typewriter 0 --share_link 0 --load_lora
else
python tools/chat_server.py --model_path $BASE_MODEL --lora_path $LORA_PATH --use_local $USE_LOCAL\
    --use_typewriter 0 --share_link 0 --load_lora
fi
