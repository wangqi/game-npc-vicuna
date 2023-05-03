#!/bin/sh

#./tools/llama -ins --repeat_penalty 4 -m models/game_npc_vicuna_gptq/game_npc_vicuna_ggml_f16.bin -i
./tools/llama -ins --repeat_penalty 4 -m models/game_npc_vicuna_gptq/game_npc_vicuna_ggml_q8_0.bin -i
