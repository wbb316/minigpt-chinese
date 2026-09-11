#!/bin/bash
# 4 组 tokenizer sample-size 实验并行启动（云端 256 核，BPE 训练单线程 → 完全并行）
# 4M 已在跑（单独启动），这里启动 8M/16M/32M
cd /root
CORPUS=/root/autodl-tmp/data/train_webnovel_v2.txt
PY=/root/miniconda3/bin/python
mkdir -p /root/tok_exp/tok /root/tok_exp/log

for S in 8000000 16000000 32000000; do
  case $S in
    8000000) TAG=8M ;;
    16000000) TAG=16M ;;
    32000000) TAG=32M ;;
  esac
  if [ -f "/root/tok_exp/tok/tok_${TAG}.pkl" ]; then
    echo "skip $TAG (已完成)"
    continue
  fi
  nohup $PY -u scratch/tok_sample_experiment.py \
    --sample $S --corpus $CORPUS \
    --out /root/tok_exp/tok/tok_${TAG}.pkl \
    > /root/tok_exp/log/train_${TAG}.log 2>&1 &
  echo "launched $TAG pid=$!"
done
echo "并行启动完成"
sleep 3
ps -eo pid,etime,cmd | grep tok_sample | grep -v grep
