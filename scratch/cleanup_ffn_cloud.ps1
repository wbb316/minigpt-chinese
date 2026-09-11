# 清理云端 FFN 消融实验产物（result_ffn_*，约 4GB）
#
# 策略：先只把**小文件**（CSV/config/日志/PNG，排除 *.pt）打包下载留档，
#       确认拿到后再 rm -rf 整个目录。
# 理由：这 5 个实验的产物云端是独一份（本地 result/ 里没有任何 FFN 归档），
#       .pt 检查点占 4GB 且价值已体现在结论里；但 val/step 曲线数据只有几十 KB，
#       丢了就再也画不出那三条对照曲线了。
. (Join-Path $PSScriptRoot 'cloud_env.ps1')
$repo = 'D:\WBB_Python\pytorch'
$dest = Join-Path $repo 'result\FFN激活消融_v3'
New-Item -ItemType Directory -Force -Path $dest | Out-Null

function Remote([string]$c) { & ssh @CloudSshOpts -p $CloudPort $CloudTarget $c 2>&1 }

$dirs = 'result_ffn_relu result_ffn_gelu result_ffn_scratch_relu result_ffn_scratch_gelu result_ffn_scratch_swiglu'

Write-Host "[1/4] 清理前：各目录大小与内容"
Remote "cd /root && for d in $dirs; do echo \"--- \$d\"; du -sh \$d 2>/dev/null; ls -1 \$d 2>/dev/null | head -8; done; echo; df -h / | tail -1" |
    ForEach-Object { Write-Host "  $_" }

Write-Host "`n[2/4] 打包小文件（排除 *.pt）"
Remote "cd /root && tar czf /tmp/ffn_small.tar.gz --exclude='*.pt' $dirs 2>/dev/null; ls -lh /tmp/ffn_small.tar.gz" |
    ForEach-Object { Write-Host "  $_" }

Write-Host "`n[3/4] 下载留档"
$tarLocal = Join-Path $dest 'ffn_variants_small_artifacts.tar.gz'
& scp @CloudSshOpts -P $CloudPort "${CloudTarget}:/tmp/ffn_small.tar.gz" $tarLocal 2>&1 | Out-Null
if (-not (Test-Path $tarLocal) -or (Get-Item $tarLocal).Length -lt 1000) {
    Write-Host "  ❌ 下载失败，**中止删除**（不能没留档就删）"
    exit 1
}
Write-Host ("  ✅ {0}  {1:N0} KB" -f (Split-Path $tarLocal -Leaf), ((Get-Item $tarLocal).Length / 1KB))

Write-Host "`n[4/4] 删除云端 FFN 目录"
Remote "cd /root && rm -rf $dirs /tmp/ffn_small.tar.gz && echo '已删除'; echo '--- 剩余 result_* ---'; ls -d /root/result_* 2>/dev/null; echo '--- 磁盘 ---'; df -h / | tail -1" |
    ForEach-Object { Write-Host "  $_" }
