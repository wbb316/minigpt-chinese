#!/bin/bash
# 本地 CPU 冒烟：6M + 新 LR schedule 各跑 8 步，验证代码路径 + 测本地速度
# 用小数据（百合 train.txt 截断到 ~2MB）避免长编码
set -e
cd /root 2>/dev/null || cd /d/WBB_Python/pytorch 2>/dev/null || cd ~/minigpt 2>/dev/null || true
echo "注意：本脚本需在项目根目录跑（train/train.py 相对路径）"
