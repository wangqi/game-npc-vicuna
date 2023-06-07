#!/bin/bash

echo "Create dir $1"
mkdir -p $1

echo "Download model $1 to models/$1"
python tools/download-model.py --threads 4 --output models/ $1
