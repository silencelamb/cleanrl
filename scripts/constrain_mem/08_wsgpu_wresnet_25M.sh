#!/bin/bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false


# 定义硬件类型
declare -a hardware_types=("wsgpu" "dojo")

# 定义模型类型和对应的模型大小
declare -A model_sizes
model_sizes["gpt"]="125M 350M 760M 1.3B 2.6B 6.7B 15B 39B 76B"
model_sizes["bert"]="Tiny Mini Small Medium Base Large LL LLL LLLL"
model_sizes["wresnet"]="25.56M 44.55M 60.19M 68.88M 126.88M"

cd ..

python cleanrl/ppo_wsc_map_cnn.py \
    --hardware wsgpu \
    --model-type wresnet \
    --model-size 25.56M \
    --gpuid 5 \
    --seed 65536 \
    --constrain-mem \
    --total-timesteps 250000 \
    --track