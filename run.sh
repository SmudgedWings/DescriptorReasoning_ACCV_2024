#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0
task_name=reason  

find_unused_port() {
    while true; do
        port=$(shuf -i 10000-60000 -n 1)
        if ! ss -tuln | grep -q ":$port "; then
            echo "$port"
            return 0
        fi
    done
}
UNUSED_PORT=$(find_unused_port)

task_id=$UNUSED_PORT

# nohup \
python reasoning/train_multigpu_reasoning.py \
    --batch_size 16 \
    --data ./datasets/h5_scannet \
    --plot_every 200 \
    --extractor_cache 'xfeat-scannet-n2048' \
    --dino_cache 'dino-scannet-dinov2_vits14' \
    -C xfeat-dinov2 \
# > ./scripts/${task_name}.log 2>&1 &

# sleep 10
# ps aux | grep 'train_multigpu_reasoning.py' | grep -v 'grep' | awk '{print $2}' > ./scripts/${task_name}.pid


# cat ./scripts/reason.pid | xargs kill