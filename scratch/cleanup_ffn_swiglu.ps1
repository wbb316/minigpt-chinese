# 补删漏掉的 result_ffn_swiglu（第一代"续训式"对照里的 swiglu 臂）
#
# 上一轮只删了 relu / gelu / scratch_{relu,gelu,swiglu} 5 个，
# result_ffn_swiglu 因之前被 head -10 截断而漏看。
. (Join-Path $PSScriptRoot 'cloud_env.ps1')
$repo = 'D:\WBB_Python\pytorch'
$dest = Join-Path $repo 'result\FFN激活消融_v3'

function Remote([string]$c) { & ssh @CloudSshOpts -p $CloudPort $CloudTarget $c 2>&1 }

Write-Host "[1/3] 当前所有 ffn 相关目录"
Remote "ls -d /root/result_ffn* /root/result_FFN* 2>/dev/null || echo '(无)'" | ForEach-Object { Write-Host "  $_" }

Write-Host "`n[2/3] 打包 result_ffn_swiglu 的小文件并下载"
Remote "cd /root && du -sh result_ffn_swiglu 2>/dev/null; ls -1 result_ffn_swiglu 2>/dev/null; tar czf /tmp/ffn_swiglu_small.tar.gz --exclude='*.pt' result_ffn_swiglu 2>/dev/null; ls -lh /tmp/ffn_swiglu_small.tar.gz" |
    ForEach-Object { Write-Host "  $_" }

$tarLocal = Join-Path $dest 'ffn_swiglu_small_artifacts.tar.gz'
& scp @CloudSshOpts -P $CloudPort "${CloudTarget}:/tmp/ffn_swiglu_small.tar.gz" $tarLocal 2>&1 | Out-Null
if (-not (Test-Path $tarLocal) -or (Get-Item $tarLocal).Length -lt 500) {
    Write-Host "  ❌ 下载失败，**中止删除**"
    exit 1
}
Write-Host ("  ✅ {0}  {1:N0} KB" -f (Split-Path $tarLocal -Leaf), ((Get-Item $tarLocal).Length / 1KB))

Write-Host "`n[3/3] 删除"
Remote "cd /root && rm -rf result_ffn_swiglu /tmp/ffn_swiglu_small.tar.gz && echo '已删除'; echo '--- 剩余 result_* ---'; ls -d /root/result_* 2>/dev/null; echo '--- 磁盘 ---'; df -h / | tail -1" |
    ForEach-Object { Write-Host "  $_" }
