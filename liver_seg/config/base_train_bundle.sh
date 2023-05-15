#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=7
export to_train=false
export to_test=true 
export testing_key="validation"
export nfolds=1

bash commands/train_bundle.sh
