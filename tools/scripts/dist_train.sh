#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

#python -m torch.distributed.launch  --nproc_per_node=${NGPUS} --master_port $PORT train.py --launcher pytorch ${PY_ARGS}
#NGPUS='4'
#PY_ARGS='/home/ubuntu2004/code/lfy/SparseKD-master/tools/cfgs/waymo_models/cp-voxel/cp-voxel-xs_sparsekd.yaml'
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --master_port $PORT --nproc_per_node=${NGPUS} train.py --launcher pytorch --cfg_file ${PY_ARGS}
