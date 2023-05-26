#!/bin/bash

echo "Clone game-npc-vicuna"
git clone https://github.com/wangqi/game-npc-vicuna.git

echo "Install Anaconda"
wget "https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh"

cd game-npc-vicuna
conda create -n gamenpc python=3.10
conda activate gamenpc

pip install -r ./requirements.txt
