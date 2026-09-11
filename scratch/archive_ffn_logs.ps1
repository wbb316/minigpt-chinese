# 补档：FFN 实验的真正记录在 /root/log/（result_ffn_* 里只有 .pt + tokenizer）
#
# 只拉「有信息量」的小文件：
#   - 两个 driver log（含每次 run 的 val 汇总，能重建三条对照曲线）
#   - 一个本地没有的脚本 run_ffn_variants_cloud.sh
# 不拉 6 个 train_ffn_*.log（每个 7MB 全是 tqdm 进度条，信息与 driver log 重复）
. (Join-Path $PSScriptRoot 'cloud_env.ps1')
$repo = 'D:\WBB_Python\pytorch'
$dest = Join-Path $repo 'result\FFN激活消融_v3'
New-Item -ItemType Directory -Force -Path $dest | Out-Null
function Remote([string]$c) { & ssh @CloudSshOpts -p $CloudPort $CloudTarget $c 2>&1 }

Write-Host "[1/3] 拉取 driver log 与云端独有脚本"
$items = @(
    @{ r = '/root/log/ffn_scratch_driver.log';            l = 'ffn_scratch_driver.log' },
    @{ r = '/root/log/ffn_variants_driver.log';           l = 'ffn_variants_driver.log' },
    @{ r = '/root/scratch/run_ffn_variants_cloud.sh';     l = 'run_ffn_variants_cloud.sh' },
    @{ r = '/root/scratch/run_ffn_scratch_cloud.sh';      l = 'run_ffn_scratch_cloud.sh' },
    @{ r = '/root/scratch/_ffn_smoke_swiglu.log';         l = 'ffn_smoke_swiglu.log' }
)
foreach ($m in $items) {
    $out = Join-Path $dest $m.l
    & scp @CloudSshOpts -P $CloudPort ("${CloudTarget}:$($m.r)") $out 2>&1 | Out-Null
    if (Test-Path $out) { Write-Host ("  ✅ {0}  {1:N0} KB" -f $m.l, ((Get-Item $out).Length/1KB)) }
    else { Write-Host ("  ⚠️ {0} 不存在" -f $m.l) }
}

Write-Host "`n[2/3] 从 driver log 提取结论（从零对照）"
Remote "grep -a -E '参数量|===== step|训练结束|最佳' /root/log/ffn_scratch_driver.log | tail -30" |
    ForEach-Object { Write-Host "  $_" }

Write-Host "`n[3/3] 本地归档目录"
Get-ChildItem $dest | Select-Object Name, @{n='KB';e={[math]::Round($_.Length/1KB,1)}} | Format-Table -AutoSize
