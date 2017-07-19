#!/bin/bash

lr=0.02
q_lambda=0.0001
subspace_num=4
gl_loss="D" # D, cosine
T=0.1
K=3
gl_lambda=0.0 # 0.1
dataset=cifar10 # cifar10, nuswide_81
log_dir=tflog_T_K_LAMBDA_

if [ -z "$1" ]; then
    gpu=0
else
    gpu=$1
fi

filename="lr_${lr}_cqlambda_${q_lambda}_subspace_num_${subspace_num}_T_${T}_K_${K}_graph_laplacian_lambda_${gl_lambda}_gl_loss_${gl_loss}_dataset_${dataset}"
model_file="models/${filename}.npy"
export TF_CPP_MIN_LOG_LEVEL=3
#                               lr  output  iter    q_lamb      n_sub   gl_loss     dataset     gpu    T       K   gl_lambda    log_dir
python train_val_script.py      $lr 300     500    $q_lambda   4       $gl_loss    $dataset    $gpu   $T     $K   $gl_lambda   $log_dir
