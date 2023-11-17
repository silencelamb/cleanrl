#!/bin/bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false


# 定义硬件类型
declare -a hardware_types=("wsgpu" "dojo")

# 定义模型类型和对应的模型大小
declare -A model_sizes
model_sizes["gpt"]="125M 350M 760M 1.3B 2.6B 6.7B"
# model_sizes["bert"]="Base Large LL LLL LLLL LLLLL"
# model_sizes["wresnet"]="25.56M 44.55M 60.19M 68.88M 126.88M"

# 定义每个模型大小对应的 micro-batchsize
declare -A batchsizes
# gpt
batchsizes["125M"]=6
batchsizes["350M"]=6
batchsizes["760M"]=1
batchsizes["1.3B"]=1
batchsizes["2.6B"]=1
batchsizes["6.7B"]=1
batchsizes["15B"]=1
batchsizes["Base"]=6
batchsizes["Large"]=6
batchsizes["LL"]=1
batchsizes["LLL"]=1
batchsizes["LLLL"]=1
batchsizes["25.56M"]=20
batchsizes["44.55M"]=18
batchsizes["60.19M"]=16
batchsizes["68.88M"]=14
batchsizes["126.88M"]=12

# 设置gpu id
declare -A gpu_ids
gpu_ids["125M"]=0
gpu_ids["350M"]=1
gpu_ids["760M"]=2
gpu_ids["1.3B"]=3
gpu_ids["2.6B"]=4
gpu_ids["6.7B"]=5
gpu_ids["15B"]=6
gpu_ids["Base"]=7
gpu_ids["Large"]=0
gpu_ids["LL"]=1
gpu_ids["LLL"]=2
gpu_ids["LLLL"]=3
gpu_ids["LLLLL"]=6
gpu_ids["25.56M"]=4
gpu_ids["44.55M"]=5
gpu_ids["60.19M"]=6
gpu_ids["68.88M"]=7
gpu_ids["126.88M"]=7
cd ..
# 循环遍历模型类型和模型大小
for model in "${!model_sizes[@]}"; do
    for size in ${model_sizes[$model]}; do
        # 获取对应的 micro-batchsize
        micro_batchsize=${batchsizes[$size]}
        gpu_id=${gpu_ids[$size]}
        echo $micro_batchsize
        # 调用脚本
        python cleanrl/ppo_wsc_map_cnn.py \
            --hardware wsgpu \
            --model-type "$model" \
            --model-size "$size" \
            --gpuid $gpu_id \
            --micro-batchsize ${micro_batchsize} \
            --rst-folder RL_1117/gpt \
            --constrain-mem \
            --seed 65536 \
            --total-timesteps 250000 \
            --track &
    done
done
