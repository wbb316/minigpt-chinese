# 跨实例拉取语料（全程后台，不阻塞）
#
#   1) 上传拉取脚本到 westc
#   2) nohup 启动 rsync 拉取（可续传）
#   3) 轮询进度直到 PULL_DONE，最后核对大小

. (Join-Path $PSScriptRoot 'cloud_env.ps1')
$repo = 'D:\WBB_Python\pytorch'
function Remote([string]$c) { & ssh @CloudSshOpts -p $CloudPort $CloudTarget $c 2>&1 }

Write-Host "[1/3] 上传拉取脚本 ..."
Remote "mkdir -p /root/log_prep /root/autodl-tmp/data" | Out-Null
& scp @CloudSshOpts -P $CloudPort (Join-Path $repo 'scratch\cloud_pull_from_westb.sh') `
     "${CloudTarget}:/root/scratch/cloud_pull_from_westb.sh" 2>&1 | Out-Null

Write-Host "[2/3] 是否已有拉取进程 ..."
$running = (Remote "pgrep -f cloud_pull_from_westb | wc -l") -join ''
if ($running.Trim() -eq '0') {
    Remote "cd /root && setsid nohup bash scratch/cloud_pull_from_westb.sh > /root/log_prep/pull.log 2>&1 < /dev/null & sleep 3; echo LAUNCHED" |
        ForEach-Object { Write-Host "  $_" }
} else {
    Write-Host "  已有进程在跑，跳过启动"
}

Write-Host "[3/3] 轮询进度（每 90 秒）..."
$deadline = (Get-Date).AddMinutes(120)
$done = $false
while ((Get-Date) -lt $deadline) {
    $sz = Remote "cd /root/autodl-tmp/data 2>/dev/null && ls -l 2>/dev/null | awk '{print `$5, `$9}'"
    $log = Remote "tail -c 300 /root/log_prep/pull.log 2>/dev/null"
    $txt = ($sz -join ' | ')
    Write-Host ("  {0}  {1}" -f (Get-Date -Format 'HH:mm:ss'), $txt)
    if (($log -join ' ') -match 'PULL_DONE') { $done = $true; break }
    Start-Sleep -Seconds 90
}

Write-Host ""
if ($done) { Write-Host "✅ 拉取完成" } else { Write-Host "⚠️ 超时，可能仍在跑（重跑本脚本可续传）" }
Write-Host "--- 最终状态 ---"
Remote "ls -l /root/autodl-tmp/data/; du -sh /root/autodl-tmp/data/*/ 2>/dev/null; df -h /root/autodl-tmp | tail -1" |
    ForEach-Object { Write-Host "  $_" }
