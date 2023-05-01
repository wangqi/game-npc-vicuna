# Assign the first command line argument to a variable or use a default value
BASE_MODEL="${1:-"huggyllama/llama-7b"}"
LORA_WEIGHT="${2:-"Facico/Chinese-Vicuna-lora-7b-3epoch-belle-and-guanaco"}"
MODEL_PATH="${3:-"models/game_npc_vicuna_base"}"
echo "The BASE_MODEL is: $BASE_MODEL"
echo "The LORA_WEIGHT is: $LORA_WEIGHT"
echo "The output model path is: $MODEL_PATH"

LORA_TOKEN="config"

echo "Merge the base '$BASE_MODEL' with '$LORA_WEIGHT' LoRA weight"
CUDA_VISIBLE_DEVICES=-1 python tools/merge.py --base_model $BASE_MODEL --lora_model $LORA_WEIGHT \
  --lora_token $LORA_TOKEN --output_dir $MODEL_PATH
