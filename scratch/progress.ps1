# 100M 正式 run 进度速查（一条命令看全部）
#
# 用法: cd D:\WBB_Python\pytorch; .\scratch\progress.ps1

. (Join-Path $PSScriptRoot 'cloud_env.ps1')
function Remote([string]$c) { & ssh @CloudSshOpts -p $CloudPort $CloudTarget $c 2>&1 }

$TOTAL = 59886

$last = (Remote "tail -1 /root/log_100m/step_history_train_webnovel_shard12.csv 2>/dev/null") -join ''
$vals = Remote "cat /root/log_100m/val_history_train_webnovel_shard12.csv 2>/dev/null"
$run  = Remote "grep -a -c '100M [full] 结束' /root/log_100m/driver.log 2>/dev/null"

if ($last -match 'Connection refused|timed out') {
    Write-Host "实例已不可连 —— 多半是训练跑完自动关机了。"
    Write-Host "重新开机后把新 SSH 地址告诉我。"
    exit 0
}

Write-Host "=================== 100M + 2B 进度 ==================="
if ($last.Trim()) {
    $f = $last.Trim() -split ','
    # CSV: step,epoch,tokens_seen,train_loss,val_loss,lr,tokens_per_sec,time
    $step = [int]$f[0]
    $pct  = [math]::Round(100.0 * $step / $TOTAL, 2)
    $seen = [double]$f[2]
    Write-Host ("进度   : {0:N0} / {1:N0} 步  ({2}%)" -f $step, $TOTAL, $pct)
    Write-Host ("已用   : {0:N2}B token" -f ($seen / 1e9))
    Write-Host ("train  : {0}" -f $f[3])
    Write-Host ("lr     : {0}" -f $f[5])
    Write-Host ("吞吐   : {0:N0} token/s" -f [double]$f[6])
    Write-Host ("时间   : {0}" -f $f[7])
    $tps = [double]$f[6]
    if ($tps -le 0) { $tps = 126000 }
    # 剩余 token = 剩余步数 × 每步 token（batch64 × block512 = 32768）
    $remain = ($TOTAL - $step) * 32768.0 / $tps
    Write-Host ("剩余约 : {0:N1} 小时（预计 {1} 完成）" -f ($remain / 3600),
                (Get-Date).AddSeconds($remain).ToString('HH:mm'))
} else {
    Write-Host "还没产生 step 记录（可能在编译/编码阶段）"
}

Write-Host "`n--- 验证点（每 5000 步一条）---"
if ($vals) { $vals | ForEach-Object { Write-Host "  $_" } } else { Write-Host "  (还没有，第一个验证点在第 5000 步 ≈ 22 分钟后)" }

if ($run.Trim() -match '^[1-9]') { Write-Host "`n✅ run 已结束" } else { Write-Host "`n🏃 运行中" }
