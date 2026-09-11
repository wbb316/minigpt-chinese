# 云端连接信息读取器 —— 各脚本 dot-source 本文件即可
#
# 为什么要有它：AutoDL 每次重新开机 SSH 端口都可能变。
# 以前每个脚本各自硬编码端口，换一次要改一堆文件；现在只改
# .autodl_key/connection.txt 里的 port 一行。
#
# 用法:
#   . (Join-Path $PSScriptRoot 'cloud_env.ps1')
#   & ssh @CloudSshOpts -p $CloudPort $CloudTarget "命令"
#   & scp @CloudSshOpts -P $CloudPort <本地> "${CloudTarget}:<远端>"

$__cfgPath = Join-Path (Split-Path $PSScriptRoot -Parent) '.autodl_key\connection.txt'
if (-not (Test-Path $__cfgPath)) { throw "找不到连接配置: $__cfgPath" }

$__cfg = @{}
foreach ($line in Get-Content $__cfgPath) {
    $s = $line.Trim()
    if (-not $s -or $s.StartsWith('#')) { continue }
    $i = $s.IndexOf('=')
    if ($i -lt 1) { continue }
    $__cfg[$s.Substring(0, $i).Trim()] = $s.Substring($i + 1).Trim()
}

foreach ($need in @('host', 'port', 'key', 'user')) {
    if (-not $__cfg.ContainsKey($need)) { throw "connection.txt 缺字段: $need" }
}
if (-not (Test-Path $__cfg['key'])) { throw "私钥不存在: $($__cfg['key'])" }

$CloudHost   = $__cfg['host']
$CloudPort   = [int]$__cfg['port']
$CloudUser   = $__cfg['user']
$CloudKey    = $__cfg['key']
$CloudTarget = "$CloudUser@$CloudHost"
# 注意：scp 的端口参数是 -P（大写），ssh 是 -p（小写）—— 混用会静默出错
$CloudSshOpts = @('-i', $CloudKey, '-o', 'StrictHostKeyChecking=no',
                  '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20')
$CloudData    = if ($__cfg.ContainsKey('remote_data')) { $__cfg['remote_data'] } else { '/root/autodl-tmp/data' }
$CloudScratch = if ($__cfg.ContainsKey('remote_scratch')) { $__cfg['remote_scratch'] } else { '/root/scratch' }
