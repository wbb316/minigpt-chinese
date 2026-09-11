# 云端状态速查（通用版，连接信息读自 .autodl_key/connection.txt）
#
# 用法: & .\scratch\cloud_status.ps1
#
# 注：所有远端命令**不含 $() 和内层引号** —— PowerShell→ssh 传参会吃掉/展开它们，
#     今天已经踩过三次（tr "\n"、pkill -f 自匹配、stat $()）。

. (Join-Path $PSScriptRoot 'cloud_env.ps1')

function Remote([string]$c) { & ssh @CloudSshOpts -p $CloudPort $CloudTarget $c 2>&1 }

Write-Host "=== 时间 / 主机 ==="
Remote "date '+%F %T'; hostname"

Write-Host "`n=== GPU ==="
Remote "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader"

Write-Host "`n=== 进程 ==="
Remote "echo -n 'python 进程数: '; pgrep -c python || echo 0; pgrep -a python | head -1 | cut -c1-110"

Write-Host "`n=== 磁盘 ==="
Remote "df -h /root/autodl-tmp / | tail -2"

Write-Host "`n=== 语料 ==="
Remote "ls -lh /root/autodl-tmp/data/*.txt 2>/dev/null || echo '(无语料)'"

Write-Host "`n=== 缓存目录占用 ==="
Remote "du -sh /root/autodl-tmp/data/*/ 2>/dev/null | tail -8 || echo '(无缓存)'"

Write-Host "`n=== 上传临时目录 ==="
Remote "du -sh /root/autodl-tmp/upload 2>/dev/null || echo '(无)'"

Write-Host "`n=== 训练进度（读 step CSV，避免 tqdm 进度条的 \r 把日志刷屏）==="
foreach ($d in @('/root/log_100m', '/root/log_100m_smoke')) {
    Write-Host "--- $d ---"
    Remote "ls $d 2>/dev/null | head -5; f=$d/step_history_train_webnovel_shard12.csv; if [ -f \$f ]; then echo -n '  最后一行: '; tail -1 \$f; echo -n '  总行数: '; wc -l < \$f; fi" |
        ForEach-Object { Write-Host "  $_" }
}

Write-Host "`n=== 验证点 ==="
foreach ($d in @('/root/log_100m', '/root/log_100m_smoke')) {
    Remote "cat $d/val_history_train_webnovel_shard12.csv 2>/dev/null | tail -6 || echo '(无)'" |
        ForEach-Object { Write-Host "  $_" }
}

Write-Host "`n=== 编码日志 ==="
Remote "tail -12 /root/log_prep/encode.log 2>/dev/null || echo '(无)'" | ForEach-Object { Write-Host "  $_" }
