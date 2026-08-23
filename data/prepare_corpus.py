"""清洗中文语料 + 按文件划分训练/验证集。

读取 D:/百合 文件夹下全部 txt，
清洗后：每本书前90% → train.txt，后10% → val.txt。
这样验证集覆盖所有书的尾部，训练集覆盖所有书的前部（更公平）。
"""
import re
import os


def clean_text(text: str) -> str:
    """清洗原始文本。"""
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        line = line.strip()                       # 去首尾空白
        if not line:                               # 空行跳过
            continue
        # 删广告行（含网址/文库字样）
        if re.search(r'(www\.|文库|轻之国度|录入|图源|扫图|修图|天使动漫)', line):
            continue
        cleaned.append(line)
    return '\n'.join(cleaned)


def read_file_with_encoding(path):
    """自动检测编码读取文本。"""
    for enc in ['gbk', 'gb18030', 'utf-8']:
        try:
            with open(path, encoding=enc) as f:
                return f.read()
        except Exception:
            continue
    return None


if __name__ == '__main__':
    corpus_dir = 'D:/百合'
    files = sorted([f for f in os.listdir(corpus_dir) if f.endswith('.txt')])
    print(f'找到 {len(files)} 本小说')

    train_parts, val_parts = [], []
    all_raw, all_cleaned = 0, 0

    for fname in files:
        path = os.path.join(corpus_dir, fname)
        raw = read_file_with_encoding(path)
        if raw is None:
            print(f'  ⚠️ {fname}: 无法识别编码，跳过')
            continue
        cleaned = clean_text(raw)
        # ★ 每本书：前90%训练，后10%验证
        split_point = int(len(cleaned) * 0.9)
        train_parts.append(cleaned[:split_point])
        val_parts.append(cleaned[split_point:])
        print(f'  {fname}: {len(raw)} → {len(cleaned)} 字符 (训{len(cleaned[:split_point])}/验{len(cleaned[split_point:])})')
        all_raw += len(raw)
        all_cleaned += len(cleaned)

    # 生成两个文件：train.txt + val.txt
    train_text = '\n'.join(train_parts)
    val_text = '\n'.join(val_parts)

    train_path = 'D:/WBB_Python/pytorch/data/train.txt'
    val_path = 'D:/WBB_Python/pytorch/data/val.txt'
    with open(train_path, 'w', encoding='utf-8') as f:
        f.write(train_text)
    with open(val_path, 'w', encoding='utf-8') as f:
        f.write(val_text)

    print(f'完成: {all_raw} → {all_cleaned} 字符')
    print(f'训练集: {len(train_text)} 字符 → {train_path}')
    print(f'验证集: {len(val_text)} 字符 → {val_path}')
