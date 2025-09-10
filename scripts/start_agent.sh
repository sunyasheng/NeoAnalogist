#!/bin/bash

# usage: WORK_DIR=workspace/20250722_171636 bash scripts/start_agent.sh paperbench
TASK_TYPE=${1:-"paperbench"}  # 默认为 paperbench
WORK_DIR=${WORK_DIR:-""} 


if [ "$TASK_TYPE" = "paperbench" ]; then
    WHITELIST=(
        "fre"
        "all-in-one"
        "stay-on-topic-with-classifier-free-guidance"
        # "semantic-self-consistency"
        # "lbcs"
        # "pinn"
        # "lca-on-the-line"
        # "self-expansion"
        # "adaptive-pruning"
        # "bam"
        # "ftrl"
        # "sample-specific-masks"
    )
    for paper_dir in debug/data/*/; do
        if [ -d "$paper_dir" ]; then
            paper_id=$(basename "$paper_dir")            
            is_whitelisted=false
            for paper in "${WHITELIST[@]}"; do
                if [ "$paper" = "$paper_id" ]; then
                    is_whitelisted=true
                    break
                fi
            done

            if [ "$is_whitelisted" = true ]; then
                paper_path="debug/data/$paper_id"
                echo "Processing whitelisted paper: $paper_id"
                if [ -n "$WORK_DIR" ]; then
                    echo "Resuming from work dir: $WORK_DIR"
                    PYTHONPATH=./ python main.py --config config/paperbench_config_w_condenser.json --task "reproduce a paper for me" --max-steps 300 --task-type paperbench --paper-path "$paper_path" --work-dir "$WORK_DIR"
                else
                    PYTHONPATH=./ python main.py --config config/paperbench_config_w_condenser.json --task "reproduce a paper for me" --max-steps 300 --task-type paperbench --paper-path "$paper_path"
                fi
            else
                echo "Skipping non-whitelisted paper: $paper_id"
            fi
        fi
    done
elif [ "$TASK_TYPE" = "devai" ]; then
    for instance_file in evaluation/devai/dataset/instances/*; do
        echo "instance_file: $instance_file"
        if [ -f "$instance_file" ]; then
            instance_path="evaluation/devai/dataset/instances/$(basename "$instance_file")"
            echo "Processing devai instance: $instance_path"
            if [ -n "$WORK_DIR" ]; then
                echo "Resuming from work dir: $WORK_DIR"
                PYTHONPATH=./ python main.py --config config/paperbench_config_w_condenser.json --task "complete the development task" --max-steps 300 --task-type devai --devai-path "$instance_path" --work-dir "$WORK_DIR"
            else
                PYTHONPATH=./ python main.py --config config/paperbench_config_w_condenser.json --task "complete the development task" --max-steps 300 --task-type devai --devai-path "$instance_path"
            fi
        fi
        break
    done
else
    echo "Unsupported task type: $TASK_TYPE"
    exit 1
fi
