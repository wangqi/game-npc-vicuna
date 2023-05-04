#BASE_MODEL="models/huggyllama_chinese-vicuna-7b-3epock-belle-guanaco/"
#BASE_MODEL="decapoda-research/llama-7b-hf"
#LORA_PATH="Chinese-Vicuna/Chinese-Vicuna-lora-7b-belle-and-guanaco"
#LORA_PATH="Facico/Chinese-Vicuna-lora-7b-3epoch-belle-and-guanaco"
#BASE_MODEL="models/game_npc_vicuna_base"
#BASE_MODEL="models/huggyllama_chinese-vicuna-7b-3epock-belle-guanaco-chinese-llama-alpaca"
#LORA_PATH="lora_out/final" #"checkpoint-6000"

MODEL_TOKEN="config"

echo ""
echo ""
echo "------------------------------------------------------"
BASE_MODEL="models/huggyllama_chinese-alpaca-plus-lora-7b-vicuna"
echo "BASE_MODEL: $BASE_MODEL; LORA_MODEL: "
python test/test.py --base_model $BASE_MODEL --model_token $BASE_MODEL

echo ""
echo ""
echo "------------------------------------------------------"
BASE_MODEL="models/huggyllama_chinese-alpaca-plus-lora-7b"
echo "BASE_MODEL: $BASE_MODEL; LORA_MODEL: "
python test/test.py --base_model $BASE_MODEL --model_token $BASE_MODEL
exit(0)

echo ""
echo ""
echo "------------------------------------------------------"
BASE_MODEL="huggyllama/llama-7b"
LORA_MODEL="ziqingyang/chinese-alpaca-plus-lora-7b"
echo "BASE_MODEL: $BASE_MODEL; LORA_MODEL: $LORA_MODEL"
python test/test.py --base_model $BASE_MODEL --lora_model $LORA_MODEL --model_token $MODEL_TOKEN --use_lora

echo ""
echo ""
echo "------------------------------------------------------"
BASE_MODEL="decapoda-research/llama-7b-hf"
LORA_MODEL="ziqingyang/chinese-alpaca-plus-lora-7b"
echo "BASE_MODEL: $BASE_MODEL; LORA_MODEL: $LORA_MODEL"
python test/test.py --base_model $BASE_MODEL --lora_model $LORA_MODEL --model_token $MODEL_TOKEN --use_lora

echo ""
echo ""
echo "------------------------------------------------------"
BASE_MODEL="huggyllama/llama-7b"
LORA_MODEL="Chinese-Vicuna/Chinese-Vicuna-lora-7b-belle-and-guanaco"
echo "BASE_MODEL: $BASE_MODEL; LORA_MODEL: $LORA_MODEL"
python test/test.py --base_model $BASE_MODEL --lora_model $LORA_MODEL --model_token $MODEL_TOKEN --use_lora

echo ""
echo ""
echo "------------------------------------------------------"
BASE_MODEL="huggyllama/llama-7b"
LORA_MODEL="Facico/Chinese-Vicuna-lora-7b-3epoch-belle-and-guanaco"
echo "BASE_MODEL: $BASE_MODEL; LORA_MODEL: $LORA_MODEL"
python test/test.py --base_model $BASE_MODEL --lora_model $LORA_MODEL --model_token $MODEL_TOKEN --use_lora

echo ""
echo ""
echo "------------------------------------------------------"
BASE_MODEL="models/huggyllama_chinese-vicuna-7b-3epock-belle-guanaco-chinese-llama-alpaca"
echo "BASE_MODEL: $BASE_MODEL; LORA_MODEL: $LORA_MODEL"
python test/test.py --base_model $BASE_MODEL --model_token $MODEL_TOKEN
