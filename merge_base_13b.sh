# Assign the first command line argument to a variable or use a default value
BASE_MODEL="${1:-"decapoda-research/llama-13b-hf"}"
LORA_WEIGHT="${2:-"ziqingyang/chinese-llama-plus-lora-13b,ziqingyang/chinese-alpaca-plus-lora-13b"}"
MODEL_PATH="${3:-"models/chinese-alpaca-plus-lora-13b"}"
OUT_TYPE="f16"
QUANT_TYPE="q4_1"

mkdir -p $MODEL_PATH
echo "The BASE_MODEL is: $BASE_MODEL"
echo "The LORA_WEIGHT is: $LORA_WEIGHT"
echo "The output model path is: $MODEL_PATH, the output type is: $OUT_TYPE"

echo "Merge the base '$BASE_MODEL' with '$LORA_WEIGHT' LoRA weight"
python tools/merge/merge_llama_with_chinese_lora.py --base_model $BASE_MODEL --lora_model $LORA_WEIGHT \
    --offload_dir /mnt/c/offload_dir/ --output_dir $MODEL_PATH

echo python tools/convert_ggml.py --outtype $OUT_TYPE --outfile $MODEL_PATH/ggml-$OUT_TYPE.bin $MODEL_PATH/consolidated.00.pth
python tools/convert_ggml.py --outtype $OUT_TYPE --outfile $MODEL_PATH/ggml-$OUT_TYPE.bin $MODEL_PATH/consolidated.00.pth

echo "quantize the model to q4_1"
echo tools/quantize $MODEL_PATH/$MODEL $MODEL_PATH/ggml_$QUANT_TYPE.bin $QUANT_TYPE
./tools/quantize $MODEL_PATH/ggml_$QUANT_TYPE.bin $MODEL_PATH/ggml_$QUANT_TYPE.bin $QUANT_TYPE
