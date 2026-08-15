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
    # 读 GBK → 清洗 → 存 UTF-8
    with open('D:/迅雷下载/2751.txt', encoding='gbk', errors='ignore') as f:
        raw = f.read()
    cleaned = clean_text(raw)
    # 存成 UTF-8 的语料文件
    with open('D:/WBB_Python/pytorch/data/corpus.txt', 'w', encoding='utf-8') as f:
        f.write(cleaned)
    print(f'清洗完成: {len(raw)} 字符 → {len(cleaned)} 字符')