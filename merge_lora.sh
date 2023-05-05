# Assign the first command line argument to a variable or use a default value
BASE_MODEL="${1:-"huggyllama/llama-7b"}"
#LORA_WEIGHT="${2:-"models/chinese_llama_plus_lora_7b,models/chinese_alpaca_plus_lora_7b"}"
#LORA_WEIGHT="${2:-"models/chinese_llama_plus_lora_7b,models/chinese_alpaca_plus_lora_7b,Facico/Chinese-Vicuna-lora-7b-3epoch-belle-and-guanaco"}"
#LORA_WEIGHT="${2:-"ziqingyang/chinese-llama-plus-lora-7b,ziqingyang/chinese-alpaca-plus-lora-7b,Facico/Chinese-Vicuna-lora-7b-3epoch-belle-and-guanaco"}"
LORA_WEIGHT="${2:-"ziqingyang/chinese-llama-plus-lora-7b,ziqingyang/chinese-alpaca-plus-lora-7b"}"
MODEL_PATH="${3:-"models/huggyllama_chinese-alpaca-plus-lora-7b-test"}"
OUT_TYPE="f16"

mkdir -p $MODEL_PATH
echo "The BASE_MODEL is: $BASE_MODEL"
echo "The LORA_WEIGHT is: $LORA_WEIGHT"
echo "The output model path is: $MODEL_PATH, the output type is: $OUT_TYPE"

echo "Merge the base '$BASE_MODEL' with '$LORA_WEIGHT' LoRA weight"
python tools/merge/merge_llama_with_chinese_lora.py --base_model $BASE_MODEL --lora_model $LORA_WEIGHT \
    --output_dir $MODEL_PATH

python tools/convert_ggml.py --outtype $OUT_TYPE --outfile $MODEL_PATH/ggml-$OUT_TYPE.bin $MODEL_PATH/consolidated.00.pth
