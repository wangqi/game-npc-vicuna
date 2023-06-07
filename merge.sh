BASE_MODEL="models/game_npc_vicuna_base"
LORA_WEIGHT="${1:-"lora_out/final/"}"
MODEL_PATH="models/game_npc_vicuna_huntress"
OUT_TYPE="f16"
QUANT_TYPE="q5_0"

#echo "Merge the game_npc_vicuna_base model with fine-tuned weight at $LORA_PATH to $MODEL_PATH"
#python tools/merge.py --base_model $BASE_MODEL --lora_model $LORA_WEIGHT \
# --lora_token $LORA_TOKEN --output_dir $MODEL_PATH

echo "Merge the base '$BASE_MODEL' with '$LORA_WEIGHT' LoRA weight"
python tools/merge/merge_llama_with_chinese_lora.py --base_model $BASE_MODEL --lora_model $LORA_WEIGHT \
    --output_dir $MODEL_PATH

echo python tools/convert_ggml.py --outtype $OUT_TYPE --outfile $MODEL_PATH/ggml-$OUT_TYPE.bin $MODEL_PATH/consolidated.00.pth
python tools/convert_ggml.py --outtype $OUT_TYPE --outfile $MODEL_PATH/ggml-$OUT_TYPE.bin $MODEL_PATH/consolidated.00.pth

echo "quantize the model ggml-$OUT_TYPE.bin $QUANT_TYPE"
echo tools/quantize $MODEL_PATH/ggml-$OUT_TYPE.bin $MODEL_PATH/ggml-$QUANT_TYPE.bin $QUANT_TYPE
tools/quantize $MODEL_PATH/ggml-$OUT_TYPE.bin $MODEL_PATH/ggml-$QUANT_TYPE.bin $QUANT_TYPE

echo "Compress the model to 4bit 128 group safetensors format. Need to install gptq.llama first"
echo "Refer to https://github.com/0cc4m/GPTQ-for-LLaMa"
echo python -m gptq.llama $MODEL_PATH c4 --wbits 4 --true-sequential --act-order --save_safetensors $MODEL_PATH/gptq-4bit-128g.safetensors
CUDA_VISIBLE_DEVICES=0 python -m gptq.llama $MODEL_PATH c4 --wbits 4 --true-sequential --act-order \
  --save_safetensors $MODEL_PATH/gptq-4bit-128g.safetensors
