#!/bin/bash

if [ -z "$OPENAI_API_KEY" ]
then
  echo "Error: OPENAI_API_KEY environment variable is not set."
  exit 1
fi
echo "OPENAI_API_KEY=$OPENAI_API_KEY"

# Rest of your script...
gradio tools/chat_server_langchain.py
