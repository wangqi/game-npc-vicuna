BASE_MODEL="models/game_npc_vicuna"
# ["f32", "f16", "q4_1", "q4_0"
OUT_TYPE="${1:-"f16"}"
OUT_PATH="models/game_npc_vicuna_gptq/"

echo "Convert the model '$BASE_MODEL' to ggml $OUT_TYPE format and save to $OUT_PATH"
mkdir -p $OUT_PATH
python tools/convert_ggml.py --outtype $OUT_TYPE --outfile $OUT_PATH/game_npc_vicuna_ggml_"$OUT_TYPE".bin $BASE_MODEL
