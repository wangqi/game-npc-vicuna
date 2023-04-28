BASE_MODEL="decapoda-research/llama-7b-hf"
LORA_PATH="Facico/Chinese-Vicuna-lora-7b-3epoch-belle-and-guanaco"
LORA_TOKEN="config"
MODEL_PATH="models/game_npc_vicuna_base"

echo "Merge the original llama-7b-hf model with lora 'Chinese-Vicuna-lora-7b-belle-and-guanaco'"
CUDA_VISIBLE_DEVICES=0 python tools/merge.py --base_model $BASE_MODEL --lora_model $LORA_PATH \
  --lora_token $LORA_TOKEN --output_dir $MODEL_PATH
