TOT_CUDA="0,1,2,3"
CUDAs=(${TOT_CUDA//,/ })
echo "CUDAs: $CUDAs"
CUDA_NUM=${#CUDAs[@]}
PORT="12345"

DATA_PATH="./data/data.json" #"../dataset/instruction/guanaco_non_chat_mini_52K-utf8.json"
OUTPUT_PATH="lora_out"
#MODEL_PATH="decapoda-research/llama-7b-hf"
MODEL_PATH="models/game_npc_vicuna_base"
# lora_checkpoint="./checkpoints/chinese-vicuna-lora-7b-belle-and-guanaco"
from_data_beginning=True
TEST_SIZE=0.3

CUDA_VISIBLE_DEVICES=${TOT_CUDA} torchrun --nproc_per_node=$CUDA_NUM --master_port=$PORT tools/finetune_f16.py \
--data_path $DATA_PATH \
--output_path $OUTPUT_PATH \
--model_path $MODEL_PATH \
--eval_steps 200 \
--save_steps 200 \
--test_size $TEST_SIZE \
#--resume_from_checkpoint $lora_checkpoint \
#--ignore_data_skip $from_data_beginning

mkdir -p $OUTPUT_PATH/final/
cp adapter_config.json $OUTPUT_PATH/final/
cp adapter_model.bin $OUTPUT_PATH/final/

