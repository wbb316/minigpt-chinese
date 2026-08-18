import re


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


if __name__ == '__main__':
    # 多本书合并清洗：读 GBK → 清洗 → 存 UTF-8
    book_files = [
        # 原 4 本
        'D:/迅雷下载/2896.txt',    # 78.8万字
        'D:/迅雷下载/2806.txt',    # 15.0万字
        'D:/迅雷下载/110290.txt',  # 8.6万字
        'D:/迅雷下载/2751.txt',    # 100万字
        # 新增 5 本
        'D:/迅雷下载/2635.txt',    # 103.8万字
        'D:/迅雷下载/3519.txt',    # 116.4万字
        'D:/迅雷下载/3585.txt',    # 54.2万字
        'D:/迅雷下载/3653.txt',    # 39.1万字
        'D:/迅雷下载/3763.txt',    # 34.6万字
    ]
    all_raw, all_cleaned = 0, 0
    parts = []
    for path in book_files:
        with open(path, encoding='gbk', errors='ignore') as f:
            raw = f.read()
        cleaned = clean_text(raw)
        parts.append(cleaned)
        print(f'{path}: {len(raw)} → {len(cleaned)} 字符')
        all_raw += len(raw)
        all_cleaned += len(cleaned)

    # 合并所有书，用分隔符隔开
    combined = '\n'.join(parts)
    with open('D:/WBB_Python/pytorch/data/corpus.txt', 'w', encoding='utf-8') as f:
        f.write(combined)
    print(f'合计: {all_raw} → {all_cleaned} 字符，已存 corpus.txt')