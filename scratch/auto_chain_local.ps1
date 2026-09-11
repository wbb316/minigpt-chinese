# 自动接力：等 4M 完成 → 跑 16M → 跑 32M（内存串行，自动衔接）
# 启动方式：作为后台 job 跑，无需人工干预
$ErrorActionPreference = 'Continue'
$root = 'D:\WBB_Python\pytorch'
Set-Location $root
$tok = "$root\tok_exp\tok"
$log = "$root\tok_exp\log"

function Wait-Tok($name) {
    while (-not (Test-Path "$tok\tok_$name.pkl")) { Start-Sleep -Seconds 20 }
    Write-Output "[$(Get-Date -Format HH:mm:ss)] tok_$name.pkl 已生成"
}

# 1) 等 4M 完成
Wait-Tok '4M'
Write-Output "[$(Get-Date -Format HH:mm:ss)] 4M 完成，等待内存释放后启动 16M"
Start-Sleep -Seconds 10

# 2) 16M（内存 ~8GB）
if (-not (Test-Path "$tok\tok_16M.pkl")) {
    Write-Output "[$(Get-Date -Format HH:mm:ss)] 启动 16M"
    python -u scratch\tok_sample_experiment.py --sample 16000000 --out "$tok\tok_16M.pkl" *>&1 | Tee-Object -FilePath "$log\train_16M.log" | Out-Null
    Write-Output "[$(Get-Date -Format HH:mm:ss)] 16M 完成"
}

# 3) 32M（内存 ~15GB，须等 16M 结束）
if (-not (Test-Path "$tok\tok_32M.pkl")) {
    Write-Output "[$(Get-Date -Format HH:mm:ss)] 启动 32M"
    python -u scratch\tok_sample_experiment.py --sample 32000000 --out "$tok\tok_32M.pkl" *>&1 | Tee-Object -FilePath "$log\train_32M.log" | Out-Null
    Write-Output "[$(Get-Date -Format HH:mm:ss)] 32M 完成"
}

New-Item -ItemType File "$root\tok_exp\ALL_DONE" -Force | Out-Null
Write-Output "[$(Get-Date -Format HH:mm:ss)] ===== 全部完成 ====="
Get-ChildItem $tok | Select-Object Name, @{n='KB';e={[math]::Round($_.Length/1KB,1)}}
Get-Content "$log\train_16M.log" -Tail 3
Get-Content "$log\train_32M.log" -Tail 3
