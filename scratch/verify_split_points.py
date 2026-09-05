# -*- coding: utf-8 -*-
"""对照验证：新 _split_points（numpy 二分）== 旧实现（暴力双重循环），逐切分点一致。

也做规模 ~20M 字符的性能对照（旧实现此时已明显慢，新实现应毫秒级）。
"""
import sys
import time

sys.path.insert(0, '.')

from data.token_cache import _split_points  # noqa: E402
from data.token_cache import _WS_RE  # noqa: E402


def _legacy_split_points(text, n_parts):
    """旧实现（修复前逻辑），逐字抄写。"""
    if n_parts <= 1:
        return []
    ws_spans = [(m.start(), m.end()) for m in _WS_RE.finditer(text)]
    if not ws_spans:
        return []
    cuts = []
    total = len(text)
    for k in range(1, n_parts):
        target = total * k // n_parts
        best = None
        for s, e in ws_spans:
            mid = (s + e) // 2
            if s <= target < e:
                best = mid
                break
            if best is None or abs(mid - target) < abs(best - target):
                best = mid
        cuts.append(best)
    cuts = sorted(set(cuts))
    return cuts


def check(text, n_parts, label):
    old = _legacy_split_points(text, n_parts)
    new = _split_points(text, n_parts)
    assert old == new, f'[{label}] n_parts={n_parts} 不一致!\nold={old}\nnew={new}'
    # 额外验证切分点都落在空白内部
    for c in new:
        assert _WS_RE.match(text, c) is not None or (
            c > 0 and _WS_RE.match(text, c - 1) is not None and text[c - 1].isspace()
        ), f'[{label}] 切分点 {c} 不在空白内部'
    print(f'  [{label}] n_parts={n_parts}: {len(new)} cuts 一致 OK')


# 各种形态文本（含密集空白/稀疏空白/无空白边界）
texts = {
    '中英混合': ('他买了 iPhone 15 和 MacBook。\n\n第二段 你好 world! \t  '
                 'x' * 30 + '\n' + '哈' * 200) * 500,
    '密集空白': ('  a  b \t\t c\n\n d  \n e   f  ' * 300),
    '中文长文': '夜 色 渐 深。' * 2000,
    '无空白': '纯中文没有空白啊亲' * 3000,
}
for label, text in texts.items():
    for n in (2, 5, 16, 64):
        check(text, n, f'{label}')

# 大文本性能对照：~20M 字符、空白区段 ~40 万
big = (('段落内容 %d：你好 world 测试。\n\n' * 40) % tuple(range(40))) * 30000
assert len(big) > 10_000_000, len(big)
t0 = time.time()
cuts_new = _split_points(big, 828)
t_new = time.time() - t0
n_ws = len(_WS_RE.findall(big))
print(f'性能对照: chars={len(big):,} ws_spans={n_ws:,} n_parts=828')
print(f'  新实现: {t_new*1000:.1f} ms, {len(cuts_new)} cuts')
print('ALL PASS')
