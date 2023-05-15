#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2
export to_train=true
export to_test=false
export nproc_per_node=3
export testing_key="validation"
export nfolds=1

bash commands/train_bundle_multi_gpu.sh
