#!/bin/bash

PROMPT=${PROMPT:-"Change the major object within the image to a pirate dog."}
WORK_DIR=${WORK_DIR:-"workspace"}
QWEN_API_URL=${QWEN_API_URL:-"http://host.docker.internal:8200"}

echo "Running agent with prompt: $PROMPT"
echo "Qwen API URL: $QWEN_API_URL"

if [ -n "$WORK_DIR" ]; then
    echo "Resuming from work dir: $WORK_DIR"
    PYTHONPATH=./ QWEN_API_URL="$QWEN_API_URL" python main.py \
        --config config/imagebrush_w_condenser.json \
        --task "$PROMPT" \
        --max-steps 20 \
        --work-dir "$WORK_DIR"
else
    PYTHONPATH=./ QWEN_API_URL="$QWEN_API_URL" python main.py \
        --config config/imagebrush_w_condenser.json \
        --task "$PROMPT" \
        --max-steps 20
fi
