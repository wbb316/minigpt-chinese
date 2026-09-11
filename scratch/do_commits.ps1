# 提交本次工作
#
# ⚠️ 只用 -F 读 UTF-8 消息文件：PowerShell 5.1 把中文当参数传给 git.exe 会按 ANSI
#    代码页转换 → commit message 变乱码。
# ⚠️ 绝不 git add docs/report_output/（豆包的目录，未跟踪状态是正常的）
$ErrorActionPreference = 'Continue'
$repo = 'D:\WBB_Python\pytorch'
Set-Location $repo
$enc = New-Object System.Text.UTF8Encoding($false)
$msgDir = Join-Path $env:TEMP 'gitmsg'
New-Item -ItemType Directory -Force -Path $msgDir | Out-Null

function Commit([string]$file, [string]$text) {
    $p = Join-Path $msgDir "$file.txt"
    [System.IO.File]::WriteAllText($p, $text, $enc)
    & git commit -F $p 2>&1 | ForEach-Object { Write-Host "  $_" }
}

# ---------- 1) 服务端：头数自动推断 + 默认模型换 100M ----------
Write-Host "[1/3] app/server.py"
& git add app/server.py
Commit 'server' @'
feat(server): 头数从 RoPE head_dim 自动推断 + 默认模型换成 100M v3_beta

- load_model 的 n_head 默认改为 None = 自动推断：RoPE 的 rope.cos_cached 形状是
  (1,1,T,head_dim)，于是 n_head = n_embd / head_dim。实测 35M→8、50M v3_alpha→9、
  100M v3_beta→11 全部推断正确；非 RoPE（旧 sinusoidal）存档没有 head_dim 信息，
  退回 8 并打印警告（如 v2 50M 是 9 头，仍需 --n-head 显式指定）
- ★ 新增防呆校验：显式传错头数时，n_embd 往往仍能被整除（704/8=88），旧代码会
  **静默加载成功但输出乱码**。现在用存档里的 head_dim 硬校验并报出应为多少头
- 默认模型：35M v2 → 100M v3_beta（16L/704d/11H, vocab 8192, swiglu+rope）
  旧模型路径写在注释里，或命令行 --ckpt/--tokenizer 覆盖
- 同时提交此前未提交的 --n-head 参数（50M 9 头支持）
- 验证：scratch/test_server_config.py（三模型推断正确 + 错误头数被拦 + 生成正常）
'@

# ---------- 2) train.py：LR scheme 扩展（此前遗留未提交）----------
Write-Host "`n[2/3] train/train.py"
& git add train/train.py
Commit 'train' @'
feat(train): LR 方案扩展到 wsd / hold-decay / const-tail

--lr-scheme 从 {cosine, const} 扩到 {cosine, const, wsd, hold-decay, const-tail}，
抽出 _cos(frac) 复用；各 scheme 的 warmup 统一取 args.warmup_steps。

依据 6M/20M-token 单遍筛选（log/6M参数+20Mtokens_LR实验/）：
S3 wsd 3.966 < S4 hold-decay 3.970 < S2 cosine-slow 4.013 < S1 cosine-fast 4.017
< S5 const-tail 4.163 —— 100M 主线 run 仍用 cosine 保持与 v3_alpha 可比。
'@

# ---------- 3) 文档 ----------
Write-Host "`n[3/3] docs（新增 yaml + 实验记录 + 状态）"
& git add docs/EXPERIMENT_LOG.md docs/MiniGPT_Project_Status.md docs/experiment_config_100M.yaml docs/tokenizer_sample_size_plan.md
Commit 'docs' @'
docs: 记录 v3 vocab 消融与 v3_beta 100M+2B；新增 experiment_config_100M.yaml

EXPERIMENT_LOG.md 追加两条：
- v3_vocab_ablation + v3_vocab_cmp_lm（2026-09-10）：样本量 4M 足够（4M→32M 仅 +0.623%
  压缩率）；vocab 消融 4096/6144/8192（前缀性质截取派生，截取的 6144 与独立训练逐位一致）；
  LM 短测各 5000 步、同一份 val → 全量 bits/char 4.1784 vs 4.1398，8192 好 0.92%，
  10/10 验证点一致 → **vocab 定版 8192**
- v3_beta_100M_ctx512_2B（2026-09-11）：16L/704d/11H / 101,457,280 参数 / shard1+2
  1.96B token 单轮 / vocab 8192 / swiglu，best val 3.0610，4.42 小时。
  ★ 明确标注为**新评估空间**（val 是 shard1+2 自己的 split）→ 不在 v2/v3 可比链内，
  不可与 v3_alpha 3.2097 排名
- 记录两个运行时坑：ulimit -n 65535（1660 分片 > 默认 1024）；容器内 shutdown 无效

MiniGPT_Project_Status.md：新增第三个评估空间、事实表加行并警示不可排名、
Current Best 补 100M 条目、已验证结论补 vocab 与方法学两条、更新下一阶段计划与 ID 规则。

experiment_config_100M.yaml：工作流第 8 步要求的本次实验 config（含实测结果与等价命令）。
tokenizer_sample_size_plan.md：tokenizer 两阶段完整计划与结果（此前未入库）。
'@

Write-Host "`n=== 提交后状态 ==="
& git log --oneline -4
Write-Host "--- 剩余未提交 ---"
& git status --short | Where-Object { $_ -notmatch 'report_output' }
