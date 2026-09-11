# 分片上传 shard12 语料（校园网 ~1 MB/s，60 分钟级别必须能断点续传）
#
#   1) 把 .gz 切成 ~190MB 分片
#   2) 逐片上传，远端大小一致则跳过（重跑本脚本即续传）
#   3) 全部到齐后在云端拼装 → 校验 gz md5 → 解压 → 校验文本字节数
#
# 用法: & .\scratch\upload_corpus.ps1

param([int]$ChunkMB = 190)

$ErrorActionPreference = 'Continue'
$PSNativeCommandUseErrorActionPreference = $false

# 连接信息统一从 .autodl_key/connection.txt 读（换端口只改那一行）
. (Join-Path $PSScriptRoot 'cloud_env.ps1')

$key  = $CloudKey
$port = $CloudPort
$t    = $CloudTarget
$o    = $CloudSshOpts
$repo = 'D:\WBB_Python\pytorch'
# 分片放临时目录（不放 repo 里，避免污染 git 状态）。
# 丢了也没关系：切分是确定性的，重跑本脚本会切出同样大小的分片，
# 再靠「远端大小一致就跳过」续传。
$chunkDir = Join-Path $env:TEMP 'shard12_upload'
$remoteUp = '/root/autodl-tmp/upload'
New-Item -ItemType Directory -Force -Path $chunkDir | Out-Null

function Remote([string]$c) { & ssh @o -p $port $t $c 2>&1 }

# name -> (gz 源文件, 原始文本文件)
$items = @(
    [pscustomobject]@{ name = 'train_webnovel_shard12'
                       gz   = Join-Path $repo 'data\train_webnovel_shard12.txt.gz'
                       txt  = Join-Path $repo 'data\train_webnovel_shard12.txt' },
    [pscustomobject]@{ name = 'val_webnovel_shard12'
                       gz   = Join-Path $repo 'data\val_webnovel_shard12.txt.gz'
                       txt  = Join-Path $repo 'data\val_webnovel_shard12.txt' }
)

Remote "mkdir -p $remoteUp /root/autodl-tmp/data" | Out-Null

$chunkBytes = $ChunkMB * 1MB
$manifest = @()

foreach ($it in $items) {
    if (-not (Test-Path $it.gz)) { Write-Host "❌ 缺 gz: $($it.gz)"; exit 1 }
    $gzLen = (Get-Item $it.gz).Length
    $nParts = [math]::Ceiling($gzLen / $chunkBytes)
    Write-Host ("=== {0}: {1:N0} 字节 → {2} 片 ===" -f $it.name, $gzLen, $nParts)

    # ---- 切分 ----
    $fs = [System.IO.File]::OpenRead($it.gz)
    for ($i = 0; $i -lt $nParts; $i++) {
        $p = Join-Path $chunkDir ("{0}.txt.gz.part{1:D2}" -f $it.name, $i)
        if ((Test-Path $p) -and (Get-Item $p).Length -gt 0) { continue }
        # 注意：gz 有 2.97GB，超过 Int32 上限 → 必须显式用 Int64，
        # 否则 [math]::Min 会选 Int32 重载并抛 "Cannot convert ... to System.Int32"
        $want = [math]::Min([int64]$chunkBytes, [int64]($gzLen - $fs.Position))
        $buf = New-Object byte[] ([int]$want)
        $got = 0
        while ($got -lt $want) {
            $r = $fs.Read($buf, $got, $want - $got)
            if ($r -le 0) { break }
            $got += $r
        }
        if ($got -eq $want) {
            [System.IO.File]::WriteAllBytes($p, $buf)   # 常见路径：不复制
        } else {
            [System.IO.File]::WriteAllBytes($p, $buf[0..([math]::Max(0, $got - 1))])
        }
    }
    $fs.Close()
    Write-Host ("  切分完成: {0} 片" -f $nParts)

    # ---- 远端已有分片的大小（一次 ssh 拿全，避免每片一次往返）----
    $listing = Remote "ls -l $remoteUp | awk '{print `$9, `$5}'"
    $have = @{}
    foreach ($ln in $listing) {
        $kv = ($ln -split '\s+')
        if ($kv.Count -ge 2 -and $kv[0] -like "*$($it.name)*part*") { $have[$kv[0]] = [int64]$kv[1] }
    }

    # ---- 逐片上传 ----
    $sent = 0; $skipped = 0
    for ($i = 0; $i -lt $nParts; $i++) {
        $p = Join-Path $chunkDir ("{0}.txt.gz.part{1:D2}" -f $it.name, $i)
        $base = Split-Path $p -Leaf
        $localLen = (Get-Item $p).Length
        if ($have.ContainsKey($base) -and $have[$base] -eq $localLen) {
            $skipped++
            continue
        }
        $sw = [Diagnostics.Stopwatch]::StartNew()
        & scp @o -P $port $p "${t}:$remoteUp/$base" 2>&1 | Out-Null
        $rc = $LASTEXITCODE
        $sw.Stop()
        if ($rc -ne 0) {
            Write-Host ("  ❌ 分片 {0} 上传失败 (rc={1})，重跑本脚本可续传" -f $base, $rc)
            exit 1
        }
        $mb = $localLen / 1MB / $sw.Elapsed.TotalSeconds
        $sent++
        Write-Host ("  [{0}/{1}] {2}  {3:N0} MB  {4:N1}s  {5:N2} MB/s" -f `
            ($i + 1), $nParts, $base, ($localLen / 1MB), $sw.Elapsed.TotalSeconds, $mb)
    }
    Write-Host ("  本文件: 新传 {0} 片, 跳过 {1} 片" -f $sent, $skipped)

    $manifest += [pscustomobject]@{
        name = $it.name
        md5  = (Get-FileHash $it.gz -Algorithm MD5).Hash.ToLower()
        txtBytes = (Get-Item $it.txt).Length
    }
}

# ---- 云端拼装 + 校验 ----
Write-Host "`n=== 云端拼装与校验 ==="
& scp @o -P $port (Join-Path $repo 'scratch\cloud_assemble.sh') "${t}:/root/scratch/cloud_assemble.sh" 2>&1 | Out-Null

foreach ($m in $manifest) {
    Write-Host ("--- {0} (期望 gz md5 {1}) ---" -f $m.name, $m.md5)
    Remote "bash /root/scratch/cloud_assemble.sh $($m.name) $($m.md5) $($m.txtBytes)" | ForEach-Object { Write-Host "  $_" }
}

Write-Host "`n=== 云端最终产物 ==="
Remote "ls -l /root/autodl-tmp/data/; df -h /root/autodl-tmp | tail -1" | ForEach-Object { Write-Host "  $_" }
