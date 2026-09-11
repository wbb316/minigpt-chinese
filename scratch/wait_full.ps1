# 等 100M 正式 run 跑完（约 4.4 小时）→ 出最终报告
#
# 全程后台。正式 run 结束时会自动关机；若已关机，ssh 会失败 —— 那本身就是
# 「跑完并关机」的信号，脚本会据此报告。
. (Join-Path $PSScriptRoot 'cloud_env.ps1')
$repo = 'D:\WBB_Python\pytorch'
function Remote([string]$c) { & ssh @CloudSshOpts -p $CloudPort $CloudTarget $c 2>&1 }

& scp @CloudSshOpts -P $CloudPort (Join-Path $repo 'scratch\cloud_full_report.sh') `
     "${CloudTarget}:/root/scratch/cloud_full_report.sh" 2>&1 | Out-Null

Write-Host "监控正式 run（每 10 分钟查一次，最多 7 小时）..."
$deadline = (Get-Date).AddHours(7)
$done = $false
while ((Get-Date) -lt $deadline) {
    $n = (Remote "grep -a -c '100M [full] 结束' /root/log_100m/driver.log 2>/dev/null") -join ''
    if ($n.Trim() -match '^[1-9]') { $done = $true; break }
    if ($n -match 'Connection refused|timed out') {
        Write-Host ("  {0}  实例已不可连（很可能已自动关机）" -f (Get-Date -Format 'HH:mm:ss'))
        $done = $true; break
    }
    $last = (Remote "tail -1 /root/log_100m/step_history_train_webnovel_shard12.csv 2>/dev/null") -join ''
    Write-Host ("  {0}  {1}" -f (Get-Date -Format 'HH:mm:ss'), $last.Trim())
    Start-Sleep -Seconds 600
}

Write-Host ""
if ($done) { Write-Host "✅ run 已结束（或实例已关机）" } else { Write-Host "⚠️ 超时" }
Write-Host ""
Remote "bash /root/scratch/cloud_full_report.sh" | ForEach-Object { Write-Host $_ }
