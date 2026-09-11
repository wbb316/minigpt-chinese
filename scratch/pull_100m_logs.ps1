# 拉取 100M+2B 正式 run 的日志，按 log\50M参数_v2+998Mtokens 的结构归档到
#   log\100M参数_v3_2Btokens\
#
# 参考目录结构（4 个文件）：
#   run_config.txt
#   step_history_train_webnovel_v2.csv      → 这里叫 _shard12
#   train_50m.log                            → 这里叫 train_100m.log
#   val_history_train_webnovel_v2.csv        → 这里叫 _shard12

. (Join-Path $PSScriptRoot 'cloud_env.ps1')
$repo = 'D:\WBB_Python\pytorch'
$dest = Join-Path $repo 'log\100M参数_v3_2Btokens'
New-Item -ItemType Directory -Force -Path $dest | Out-Null

function Remote([string]$c) { & ssh @CloudSshOpts -p $CloudPort $CloudTarget $c 2>&1 }
function Grab([string]$r, [string]$l) {
    & scp @CloudSshOpts -P $CloudPort ("${CloudTarget}:$r") $l 2>&1 | Out-Null
}

$RD = '/root/log_100m'

Write-Host "=== 云端日志目录 ==="
Remote "ls -lh $RD" | ForEach-Object { Write-Host "  $_" }

Write-Host "`n=== 拉取 ==="
$map = @(
    @{ r = "$RD/run_config.txt";                                 l = 'run_config.txt' },
    @{ r = "$RD/step_history_train_webnovel_shard12.csv";        l = 'step_history_train_webnovel_shard12.csv' },
    @{ r = "$RD/val_history_train_webnovel_shard12.csv";         l = 'val_history_train_webnovel_shard12.csv' },
    @{ r = "$RD/driver.log";                                     l = 'train_100m.log' }
)
foreach ($m in $map) {
    $out = Join-Path $dest $m.l
    Grab $m.r $out
    if (Test-Path $out) {
        Write-Host ("  ✅ {0}  {1:N1} MB" -f $m.l, ((Get-Item $out).Length / 1MB))
    } else {
        Write-Host ("  ❌ {0} 拉取失败" -f $m.l)
    }
}

Write-Host "`n=== 收尾统计（从 train_100m.log）==="
Remote "grep -a -E '训练结束|最佳模型|吞吐|显存峰值|AMP skipped|tokens_seen 合计|AUTOSHUTDOWN' $RD/driver.log | tail -10" |
    ForEach-Object { Write-Host "  $_" }

Write-Host "`n=== 归档结果: $dest ==="
Get-ChildItem $dest | Select-Object Name, @{n='MB';e={[math]::Round($_.Length/1MB,2)}} | Format-Table -AutoSize
