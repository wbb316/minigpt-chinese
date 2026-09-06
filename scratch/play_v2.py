# -*- coding: utf-8 -*-
"""v2 35M 生成体验：多 prompt × 多温度对比（主观质量评估用）。"""
import sys
sys.path.insert(0, '.')

from generate import load_model, generate

CKPT = 'result/35M参数+998Mtokens/checkpoint_best.pt'
TOK = 'result/35M参数+998Mtokens/tokenizer_best.pkl'

PROMPTS = [
    '夜色如墨，他独自站在窗前，想起三年前的那场雨。',
    '系统提示：宿主获得新手礼包，是否立即开启？',
    '多年以后，当她再次回到这座小城，',
]

TEMPS = [0.3, 0.8, 1.1]

def main():
    gpt, tokenizer = load_model(ckpt_path=CKPT, tok_path=TOK)
    for p in PROMPTS:
        print('\n' + '=' * 70)
        print(f'PROMPT: {p}')
        print('=' * 70)
        for t in TEMPS:
            out = generate(gpt, tokenizer, p, max_new_tokens=60,
                           temperature=t, top_p=0.9, repetition_penalty=1.15,
                           use_cache=True, seed=7)
            print(f'\n--- temperature={t} ---')
            print(out)

if __name__ == '__main__':
    main()
