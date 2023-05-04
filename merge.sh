BASE_MODEL="models/huggyllama_chinese-alpaca-plus-lora-7b-vicuna"
LORA_WEIGHT="${1:-"lora_out/huntress-7b"}"
MODEL_PATH="models/game_npc_vicuna_huntress"
OUT_TYPE="f16"

#echo "Merge the game_npc_vicuna_base model with fine-tuned weight at $LORA_PATH to $MODEL_PATH"
#python tools/merge.py --base_model $BASE_MODEL --lora_model $LORA_WEIGHT \
# --lora_token $LORA_TOKEN --output_dir $MODEL_PATH

echo "Merge the base '$BASE_MODEL' with '$LORA_WEIGHT' LoRA weight"
python tools/merge/merge_llama_with_chinese_lora.py --base_model $BASE_MODEL --lora_model $LORA_WEIGHT \
    --output_dir $MODEL_PATH

python tools/convert_ggml.py --outtype $OUT_TYPE --outfile $MODEL_PATH/ggml-$OUT_TYPE.bin $MODEL_PATH/consolidated.00.pth
