"""webnovel-chinese 语料工具：jsonl → 清洗 → 按真书 90/10 划分 → data/train|val_webnovel_v2.txt

来源: HuggingFace qqceqqq/webnovel-chinese（9000 本网文，~36GB jsonl，apache-2.0）
  每行一条: {"title": "书名", "chapter": "第X章 ...", "text": "正文"}
  章节按书连续排列（同书章节相邻）。

与 v0/v1 不同: 该语料有真实书名/章节且书内连续, 可以按"真书"划分——
  每本书的章节按 9:1 切成 train/val（前 90% 章节→train，后 10%→val），
  比 v0/v1 的"伪本等长块"更能测跨书泛化。

内存友好: 单遍流式 —— 读到"书结束"(title 变化)即切分落盘, 峰值内存≈单本书大小。

产物 (data/, v2 后缀避免与 v0/v1/all 混淆, 旧语料不动):
  train_webnovel_v2.txt
  val_webnovel_v2.txt

用法:
  python data/prepare_webnovel.py                    # 处理 D:\小说\webnovel\*.jsonl
  python data/prepare_webnovel.py --src DIR          # 指定 jsonl 目录
  python data/prepare_webnovel.py --shards 0 1 2     # 只处理指定分片
  python data/prepare_webnovel.py --dry              # 只统计不写文件
"""
import argparse
import codecs
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, 'data')
DEFAULT_SRC = r'D:\小说\webnovel'

# 行级水印/广告（整行丢弃）
DROP_RE = re.compile(
    r'www\.|http://|https://|书友群|QQ群|加群|群号|求月票|求推荐票|求收藏|'
    r'最新网址|请记住本书|首发|追书|下载本书|加入书签|手机用户请浏览|'
    r'最新章节全文阅读|read\.|小说阅读网|起点|纵横|创世|书旗|'
    r'未完待续|本章未完|新书推荐|推荐一本|老铁|飞卢|笔趣阁')


def clean_text(text: str) -> str:
    """正文清洗：逐行去广告水印，压缩连续空行，返回清洗后的多行文本。"""
    out = []
    for line in text.split('\n'):
        s = line.strip()
        if not s:
            continue
        if len(s) <= 3:                      # 短行多为分隔符/装饰
            continue
        if DROP_RE.search(s):
            continue
        out.append(s)
    return '\n'.join(out)


def iter_records(src_dir, shards=None):
    """流式遍历 jsonl：yield (title, chapter, text)。"""
    files = sorted(glob.glob(os.path.join(src_dir, 'webnovel_*.jsonl')))
    if shards:
        files = [f for f in files if any(f'webnovel_{i}.jsonl' in f for i in shards)]
    if not files:
        raise FileNotFoundError(f'未找到 jsonl: {src_dir}/webnovel_*.jsonl')
    for fp in files:
        print(f'📖 读取 {os.path.basename(fp)} ...', flush=True)
        with codecs.open(fp, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                title = (rec.get('title') or '').strip()
                text = (rec.get('text') or '').strip()
                if not title or not text:
                    continue
                yield title, (rec.get('chapter') or '').strip(), text


def main():
    ap = argparse.ArgumentParser(description='webnovel jsonl → 按真书 90/10 划分 train/val')
    ap.add_argument('--src', default=DEFAULT_SRC)
    ap.add_argument('--shards', type=int, nargs='*', default=None,
                    help='只处理指定分片号（默认全部）')
    ap.add_argument('--min-chars', type=int, default=30000,
                    help='少于该总字符数的书丢弃（避免残缺书）')
    ap.add_argument('--val-ratio', type=float, default=0.1)
    ap.add_argument('--dry', action='store_true', help='只统计不写文件')
    args = ap.parse_args()

    train_path = os.path.join(OUT_DIR, 'train_webnovel_v2.txt')
    val_path = os.path.join(OUT_DIR, 'val_webnovel_v2.txt')

    # ---------- 单遍流式：书结束即切分落盘 ----------
    cur_title, cur_chapters = None, []      # 正在聚合的这本书
    n_records = n_books = kept = dropped = 0
    train_chars = val_chars = 0
    train_fh = val_fh = None
    if not args.dry:
        train_fh = codecs.open(train_path, 'w', encoding='utf-8')
        val_fh = codecs.open(val_path, 'w', encoding='utf-8')
    need_sep_t = need_sep_v = False         # 书之间加空行分隔

    def flush_book(title, chapters):
        """书名变化时调用：把上一本按 90/10 切分并落盘。"""
        nonlocal n_books, kept, dropped, train_chars, val_chars
        nonlocal need_sep_t, need_sep_v
        if not chapters:
            return
        n_books += 1
        total = sum(len(c) for c in chapters)
        if total < args.min_chars:
            dropped += 1
            return
        kept += 1
        cut = max(1, int(len(chapters) * (1 - args.val_ratio)))
        train_ch = chapters[:cut]
        val_ch = chapters[cut:]
        t_text = f'【{title}】\n' + '\n\n'.join(train_ch)
        v_text = f'【{title}】\n' + '\n\n'.join(val_ch)
        train_chars += len(t_text)
        val_chars += len(v_text)
        if args.dry:
            return
        if need_sep_t:
            train_fh.write('\n\n')
        if need_sep_v and val_ch:
            val_fh.write('\n\n')
        train_fh.write(t_text)
        if val_ch:
            val_fh.write(v_text)
        need_sep_t = True
        if val_ch:
            need_sep_v = True

    try:
        for title, chapter, text in iter_records(args.src, args.shards):
            n_records += 1
            if title != cur_title:
                if cur_title is not None:
                    flush_book(cur_title, cur_chapters)
                cur_title, cur_chapters = title, []
            ct = clean_text(text)
            if ct:
                cur_chapters.append(ct)
            if n_records % 100000 == 0:
                print(f'  已读 {n_records:,} 条 / 完成 {n_books} 本书', flush=True)
        flush_book(cur_title, cur_chapters)      # 最后一本
    finally:
        if train_fh:
            train_fh.close()
        if val_fh:
            val_fh.close()

    print(f'共 {n_records:,} 条 → {n_books} 本书 | 保留 {kept} / 丢 {dropped} 本短书')
    print(f'train: {train_chars:,} 字符 | val: {val_chars:,} 字符')
    if args.dry:
        print('(dry-run，未写文件)')
        return
    print(f'✅ {train_path} ({os.path.getsize(train_path)/1e6:.1f}MB)')
    print(f'✅ {val_path} ({os.path.getsize(val_path)/1e6:.1f}MB)')


if __name__ == '__main__':
    main()
