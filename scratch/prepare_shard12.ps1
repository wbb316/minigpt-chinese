# 准备 shard1+shard2 语料（100M/2B run 用）
#
# prepare_webnovel.py 的输出路径是写死的 data/train|val_webnovel_v2.txt，
# 会覆盖现有的 shard0 语料（阶段一/tokenizer/benchmark 都依赖它）。
# 所以这里做一次「改名让路」：
#   1) shard0 文件改名备份
#   2) 跑 --shards 1 2
#   3) 产物改名成 _shard12
#   4) 无论成败都把 shard0 名字还原
#
# 用法: & .\scratch\prepare_shard12.ps1

$ErrorActionPreference = 'Continue'
$repo = 'D:\WBB_Python\pytorch'
$data = Join-Path $repo 'data'
$t  = Join-Path $data 'train_webnovel_v2.txt'
$v  = Join-Path $data 'val_webnovel_v2.txt'
$tb = Join-Path $data 'train_webnovel_v2_shard0.txt'
$vb = Join-Path $data 'val_webnovel_v2_shard0.txt'
$t12 = Join-Path $data 'train_webnovel_shard12.txt'
$v12 = Join-Path $data 'val_webnovel_shard12.txt'

Write-Host "=== 准备前状态 ==="
foreach ($p in @($t, $v, $t12, $v12)) {
    if (Test-Path $p) {
        Write-Host ("  {0}  {1:N1} MB" -f (Split-Path $p -Leaf), ((Get-Item $p).Length / 1MB))
    } else { Write-Host ("  {0}  (无)" -f (Split-Path $p -Leaf)) }
}

# ---- 1) shard0 让路 ----
$movedT = $false; $movedV = $false
if (Test-Path $t) {
    if (Test-Path $tb) { Remove-Item $tb -Force }
    Move-Item $t $tb; $movedT = $true
}
if (Test-Path $v) {
    if (Test-Path $vb) { Remove-Item $vb -Force }
    Move-Item $v $vb; $movedV = $true
}
Write-Host "已把 shard0 语料改名让路（train=$movedT val=$movedV）"

try {
    # ---- 2) 清洗 shard1 + shard2 ----
    Write-Host "=== 开始清洗 webnovel_1.jsonl + webnovel_2.jsonl ==="
    $sw = [Diagnostics.Stopwatch]::StartNew()
    & python (Join-Path $repo 'data\prepare_webnovel.py') --shards 1 2 2>&1 |
        ForEach-Object { Write-Host "  $_" }
    $rc = $LASTEXITCODE
    $sw.Stop()
    Write-Host ("清洗退出码 {0}，耗时 {1:N1} 分钟" -f $rc, $sw.Elapsed.TotalMinutes)

    # ---- 3) 产物改名成 _shard12 ----
    if (Test-Path $t) {
        if (Test-Path $t12) { Remove-Item $t12 -Force }
        Move-Item $t $t12
        Write-Host ("  ✅ train_webnovel_shard12.txt  {0:N1} MB" -f ((Get-Item $t12).Length / 1MB))
    }
    if (Test-Path $v) {
        if (Test-Path $v12) { Remove-Item $v12 -Force }
        Move-Item $v $v12
        Write-Host ("  ✅ val_webnovel_shard12.txt    {0:N1} MB" -f ((Get-Item $v12).Length / 1MB))
    }
} finally {
    # ---- 4) 还原 shard0 ----
    if ($movedT -and (Test-Path $tb)) { Move-Item $tb $t -Force }
    if ($movedV -and (Test-Path $vb)) { Move-Item $vb $v -Force }
    Write-Host "已还原 shard0 语料"
}

Write-Host "=== 结束状态 ==="
foreach ($p in @($t, $v, $t12, $v12)) {
    if (Test-Path $p) {
        Write-Host ("  {0}  {1:N1} MB" -f (Split-Path $p -Leaf), ((Get-Item $p).Length / 1MB))
    } else { Write-Host ("  {0}  (无)" -f (Split-Path $p -Leaf)) }
}
