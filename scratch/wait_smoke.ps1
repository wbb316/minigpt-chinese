# 等 100M smoke test 跑完 → 打印体检报告（全程后台）
. (Join-Path $PSScriptRoot 'cloud_env.ps1')
$repo = 'D:\WBB_Python\pytorch'
function Remote([string]$c) { & ssh @CloudSshOpts -p $CloudPort $CloudTarget $c 2>&1 }

# 上传报告脚本
& scp @CloudSshOpts -P $CloudPort (Join-Path $repo 'scratch\cloud_smoke_report.sh') `
     "${CloudTarget}:/root/scratch/cloud_smoke_report.sh" 2>&1 | Out-Null

Write-Host "等待 smoke 结束（最多 40 分钟）..."
$deadline = (Get-Date).AddMinutes(40)
$done = $false
while ((Get-Date) -lt $deadline) {
    $n = (Remote "grep -a -c '100M \[smoke\] 结束' /root/log_100m_smoke/driver.log 2>/dev/null") -join ''
    if ($n.Trim() -match '^[1-9]') { $done = $true; break }
    # 编码阶段看进度
    $enc = (Remote "tail -2 /root/log_100m_smoke/driver.log 2>/dev/null | tr '\r' '\n' | tail -1") -join ''
    Write-Host ("  {0}  {1}" -f (Get-Date -Format 'HH:mm:ss'), $enc.Substring(0, [Math]::Min(110, $enc.Length)))
    Start-Sleep -Seconds 45
}

Write-Host ""
if ($done) { Write-Host "✅ smoke 已结束" } else { Write-Host "⚠️ 超时（可能还在跑）" }
Write-Host ""
Remote "bash /root/scratch/cloud_smoke_report.sh" | ForEach-Object { Write-Host $_ }
