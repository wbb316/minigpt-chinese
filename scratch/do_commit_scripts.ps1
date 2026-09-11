# 提交工具脚本与基准数据 + .gitattributes
#
# ⚠️ 9 个含主机/端口的脚本已在 .gitignore 里排除，不会被 add。
# ⚠️ 不动 docs/report_output/（豆包的目录）。
$ErrorActionPreference = 'Continue'
$repo = 'D:\WBB_Python\pytorch'
Set-Location $repo
$enc = New-Object System.Text.UTF8Encoding($false)
$msgDir = Join-Path $env:TEMP 'gitmsg'
New-Item -ItemType Directory -Force -Path $msgDir | Out-Null

Write-Host "=== 待提交内容体量 ==="
foreach ($d in 'scratch', 'tok_bench', 'tok_exp') {
    $files = Get-ChildItem $d -Recurse -File -ErrorAction SilentlyContinue |
             Where-Object { $_.Name -notlike '*.pkl' -and $_.Name -notlike '*.pt' }
    $sum = ($files | Measure-Object Length -Sum).Sum
    Write-Host ("  {0,-10} {1,4} 个文件  {2,8:N2} MB" -f $d, $files.Count, ($sum / 1MB))
}

Write-Host "`n=== 确认被忽略的敏感脚本（应为 9 个）==="
& git status --ignored --short scratch/ 2>&1 | Select-String '^!!' | ForEach-Object { Write-Host "  $($_.Line)" }

Write-Host "`n=== add + commit ==="
& git add .gitattributes .gitignore scratch tok_bench tok_exp 2>&1 | ForEach-Object { Write-Host "  $_" }

$msg = @'
chore: 提交实验工具脚本与 tokenizer 基准数据；补 .gitattributes 强制 .sh 用 LF

.gitattributes（新增，重要）：
仓库原先没有它，而 Windows 上 core.autocrlf=true → 提交的 .sh 会在 checkout 时
被替换成 CRLF，scp 到 Linux 后 bash 报 `$'\r': command not found`。现在强制
*.sh eol=lf、*.ps1 eol=crlf、脚本/文档/数据类 LF、权重与图片类 binary。
**已有的 .sh 需要 `git add --renormalize .` 再提交一次才会被修正。**

.gitignore（新增一条安全规则）：
把 9 个把 AutoDL 主机/端口写死的一次性脚本排除在版本库外（Status 文件要求
「ssh 地址另行保存，不入库」）。长期脚本改为 dot-source scratch/cloud_env.ps1，
从 .autodl_key/connection.txt 读连接信息（该文件已在 .gitignore 中）。

scratch/：本次全部实验的工具脚本
- cloud_env.ps1 + connection.txt 约定：连接信息单一来源，换端口只改一行
- cloud_run_100m.sh / launch_100m.ps1：100M+2B run（含 ulimit -n 65535 与
  语料放系统盘抗克隆的说明）
- eval_val_bits.py：跨词表可比评测（bits/char、bits/byte），vocab 消融的判据
- derive_vocab_truncate.py / vocab_tail.py / cmp_tokenizers.py：vocab 消融三件套
- tok_sample_experiment.py / build_tok_benchmark.py / eval_tokenizers.py：样本量实验
- plot_*.py / progress*.ps1|sh / cloud_status.ps1：出图与监控
- 其余为各阶段一次性自动化（LR 筛选、FFN 消融、语料准备与上传）

tok_bench/：tokenizer 评测的固定 held-out 基准（B1/B2/B3 + meta，seed 20260910）
tok_exp/：tokenizer 两阶段的评测 JSON 与汇总图（.pkl 已被 *.pkl 忽略）
'@
$p = Join-Path $msgDir 'scripts.txt'
[System.IO.File]::WriteAllText($p, $msg, $enc)
& git commit -F $p 2>&1 | ForEach-Object { Write-Host "  $_" }

Write-Host "`n=== 结果 ==="
& git log --oneline -5
Write-Host "--- 剩余未提交（排除 report_output）---"
& git status --short | Where-Object { $_ -notmatch 'report_output' }
