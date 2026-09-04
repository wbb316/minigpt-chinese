"""轻小说语料工具：清洗 + 伪本切片 + 90/10 划分 + 版本合并。

背景: v0/v1 的 concat.txt 都被作者"部分清洗"过(书头被删/粘连), 无法可靠按真书切分,
      统一采用"伪本"方案(等长块≈单本体量), 每个伪本前90%→train, 后10%→val。

产物(都在 data/):
  train/val_lightnovel.txt         ← v0 语料(旧版默认, 不动)
  train/val_lightnovel_v1.txt      ← v1 语料(~1500 本)
  train/val_lightnovel_v0.txt      ← v0 副本(命名统一用)
  train/val_lightnovel_all.txt     ← v0+v1 合并
原 百合 语料 (data/train.txt / data/val.txt) 完全不动。

用法:
  python data/prepare_lightnovel.py v1    # 生成 v1 两件套(~1500 伪本)
  python data/prepare_lightnovel.py copy  # v0 -> _v0 后缀副本(命名统一)
  python data/prepare_lightnovel.py all   # v0+v1 合并 -> _all 两件套
"""
import codecs
import os
import re
import shutil
import sys

V0_SRC = r'D:\小说\LightNovel5000_v0_concat.txt'
V1_SRC = r'D:\小说\LightNovel5000_v1_concat.txt'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, 'data')
V0_N = 949      # v0 官方书数
V1_N = 1500     # v1 官方书数(约)

# 清洗: 整行丢弃(空白/空行/水印广告行)
DROP_RE = re.compile(r'www\.|文库|轻之国度|录入|图源|扫图|修图|天使动漫|转自|'
                     r'(?:台版|港版|日版|网译版|繁中|简中|中文版|录入组|制作组)')
CHARS_PER_TOKEN = 1.3   # 估算用


def clean_line(line: str):
    line = line.strip()
    if not line:
        return None
    if DROP_RE.search(line):
        return None
    return line


def pseudo_split(src, n_chunks, train_path, val_path):
    """读全文 → 清洗 → 切成 n 个伪本 → 各 90/10 → 写 train/val。"""
    with codecs.open(src, 'r', encoding='utf-8', errors='replace') as f:
        raw = f.read()
    print(f'原始字符: {len(raw):,}')

    cleaned_lines = []
    for line in raw.split('\n'):
        c = clean_line(line.rstrip('\r'))
        if c is not None:
            cleaned_lines.append(c)
    cleaned = '\n'.join(cleaned_lines)
    del raw
    print(f'清洗后: {len(cleaned):,} 字符 (~{len(cleaned)/CHARS_PER_TOKEN:,.0f} tokens)')

    n = max(1, n_chunks)
    chunk = max(1, len(cleaned) // n)
    train_parts, val_parts = [], []
    pos = 0
    while pos < len(cleaned):
        end = min(len(cleaned), pos + chunk)
        piece = cleaned[pos:end]
        cut = int(len(piece) * 0.9)
        train_parts.append(piece[:cut])
        val_parts.append(piece[cut:])
        pos = end
    print(f'切成 {len(train_parts)} 个伪本(每个 ~{chunk:,} 字符)')
    del cleaned

    with codecs.open(train_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(p for p in train_parts if p))
    with codecs.open(val_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(p for p in val_parts if p))
    del train_parts, val_parts
    print(f'train: {os.path.getsize(train_path)/1e6:.1f}MB → {train_path}')
    print(f'val:   {os.path.getsize(val_path)/1e6:.1f}MB → {val_path}')


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'v1'
    if cmd == 'v1':
        pseudo_split(V1_SRC, V1_N,
                     os.path.join(OUT_DIR, 'train_lightnovel_v1.txt'),
                     os.path.join(OUT_DIR, 'val_lightnovel_v1.txt'))
    elif cmd == 'copy':
        for src, dst in [('train_lightnovel.txt', 'train_lightnovel_v0.txt'),
                         ('val_lightnovel.txt', 'val_lightnovel_v0.txt')]:
            shutil.copyfile(os.path.join(OUT_DIR, src), os.path.join(OUT_DIR, dst))
            print(f'copy: {src} → {dst}')
    elif cmd == 'all':
        for a, b, out in [('train_lightnovel.txt', 'train_lightnovel_v1.txt',
                           'train_lightnovel_all.txt'),
                          ('val_lightnovel.txt', 'val_lightnovel_v1.txt',
                           'val_lightnovel_all.txt')]:
            parts = []
            for p in (a, b):
                with codecs.open(os.path.join(OUT_DIR, p), 'r', encoding='utf-8') as f:
                    parts.append(f.read())
            text = '\n'.join(parts)
            with codecs.open(os.path.join(OUT_DIR, out), 'w', encoding='utf-8') as f:
                f.write(text)
            print(f'all: {a}+{b} → {out} ({len(text):,} 字符, '
                  f'{os.path.getsize(os.path.join(OUT_DIR, out))/1e6:.1f}MB)')
    else:
        print(__doc__)


if __name__ == '__main__':
    main()
