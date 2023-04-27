DATA_PATH="./data/data.json" #"../dataset/instruction/guanaco_non_chat_mini_52K-utf8.json"
OUTPUT_PATH="lora_out"
MODEL_PATH="decapoda-research/llama-7b-hf"
lora_checkpoint="./checkpoints/checkpoint-11600"
from_data_beginning=True

python tools/finetune.py \
--data_path $DATA_PATH \
--output_path $OUTPUT_PATH \
--model_path $MODEL_PATH \
--eval_steps 200 \
--save_steps 200 \
--resume_from_checkpoint $lora_checkpoint \
--ignore_data_skip $from_data_beginning

mkdir -p $OUTPUT_PATH/final/
cp adapter_config.json $OUTPUT_PATH/final/
cp adapter_model.bin $OUTPUT_PATH/final/
