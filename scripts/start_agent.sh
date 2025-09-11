#!/bin/bash

PROMPT=${PROMPT:-"Change the major object within the image to a pirate dog."}
WORK_DIR=${WORK_DIR:-"workspace"}

echo "Running agent with prompt: $PROMPT"
if [ -n "$WORK_DIR" ]; then
    echo "Resuming from work dir: $WORK_DIR"
    PYTHONPATH=./ python main.py \
        --config config/imagebrush_w_condenser.json \
        --task "$PROMPT" \
        --max-steps 300 \
        --work-dir "$WORK_DIR"
else
    PYTHONPATH=./ python main.py \
        --config config/imagebrush_w_condenser.json \
        --task "$PROMPT" \
        --max-steps 300
fi
