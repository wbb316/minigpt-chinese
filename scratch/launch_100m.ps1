# 100M 训练启动器：预置 tokenizer + 上传脚本 + 启动（smoke / full）
#
# 用法:
#   & .\scratch\launch_100m.ps1 -Mode smoke   # 200-500 步体检
#   & .\scratch\launch_100m.ps1 -Mode full    # 正式 run
#
# 注意：clone 出来的新机 /root/autodl-tmp 是空的数据盘，但 /root（系统盘）带过来了，
# 里面有代码和 result_* 目录；tokenizer 仍需从本地上传以保证与对照实验完全同一个。

param([ValidateSet('smoke', 'full')][string]$Mode = 'smoke')

$ErrorActionPreference = 'Continue'
$PSNativeCommandUseErrorActionPreference = $false

# 连接信息统一从 .autodl_key/connection.txt 读（换端口只改那一行）
. (Join-Path $PSScriptRoot 'cloud_env.ps1')

$key  = $CloudKey
$port = $CloudPort
$t    = $CloudTarget
$o    = $CloudSshOpts
$repo = 'D:\WBB_Python\pytorch'

if ($Mode -eq 'smoke') { $logDir = '/root/log_100m_smoke' } else { $logDir = '/root/log_100m' }

function Remote([string]$c) { & ssh @o -p $port $t $c 2>&1 }

Write-Host "[1/4] 建目录 + 校验语料（语料在**系统盘** /root/data，抗克隆）..."
$chk = Remote "mkdir -p /root/scratch /root/result_vcmp/v8192 /root/autodl-tmp/data $logDir && ls -l /root/data/*shard12* /root/autodl-tmp/data/ 2>/dev/null"
$chk | ForEach-Object { Write-Host "  $_" }
$joined = $chk -join ' '
if ($joined -notmatch 'train_webnovel_shard12\.txt' -or $joined -notmatch 'val_webnovel_shard12\.txt') {
    Write-Host "❌ 语料未就位（/root/data 下缺 shard12 train/val）"
    exit 1
}
# 系统盘必须同时有语料；数据盘保持空闲以放缓存
if ($joined -notmatch '/root/data/') {
    Write-Host "❌ 语料不在 /root/data（系统盘），克隆后仍会丢"
    exit 1
}

Write-Host "[2/4] 上传 tokenizer（与 6144/8192 对照实验同一个文件）..."
& scp @o -P $port (Join-Path $repo 'tok_exp\tok\tok_v8192.pkl') `
     "${t}:/root/result_vcmp/v8192/tokenizer_v8192_s4000000.pkl" 2>&1 | Out-Null
$m1 = (Get-FileHash (Join-Path $repo 'tok_exp\tok\tok_v8192.pkl') -Algorithm MD5).Hash.ToLower()
$m2 = (Remote "md5sum /root/result_vcmp/v8192/tokenizer_v8192_s4000000.pkl | cut -d' ' -f1") -join ''
$m2 = $m2.Trim().ToLower()
Write-Host "  本地 md5: $m1"
Write-Host "  云端 md5: $m2"
if ($m1 -ne $m2) { Write-Host "❌ tokenizer md5 不一致，终止"; exit 1 }
Write-Host "  ✅ 一致"

Write-Host "[3/4] 上传训练脚本 ..."
& scp @o -P $port (Join-Path $repo 'scratch\cloud_run_100m.sh') "${t}:/root/scratch/cloud_run_100m.sh" 2>&1 | Out-Null
Write-Host "  ✅ 完成"

Write-Host "[4/4] 启动 [$Mode] ..."
Remote "cd /root && setsid nohup bash scratch/cloud_run_100m.sh $Mode > $logDir/driver.log 2>&1 < /dev/null & sleep 5; echo LAUNCHED" |
    ForEach-Object { Write-Host "  $_" }
Start-Sleep -Seconds 20
Write-Host "--- driver.log ---"
Remote "cat $logDir/driver.log 2>/dev/null | head -25" | ForEach-Object { Write-Host "  $_" }
Write-Host ""
Write-Host "监控: ssh -i `"$key`" -p $port $t `"tail -3 $logDir/driver.log`""
