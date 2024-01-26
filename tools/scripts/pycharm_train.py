#!/usr/bin/env bash
import torch
torch.distributed.launch  --nproc_per_node=${NGPUS} --master_port 29567 train.py --launcher pytorch ${PY_ARGS}
