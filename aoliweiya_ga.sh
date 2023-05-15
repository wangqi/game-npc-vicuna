#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gamenpc

cd tools/generative_agent

python ./aoliweiya_ga.py

cd -
