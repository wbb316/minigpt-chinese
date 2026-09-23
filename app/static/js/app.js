
    'use strict';
    (function () {
        // ===== 元素句柄 =====
        const $ = (id) => document.getElementById(id);
        const el = {
            subtitle: $('subtitle'), miName: $('miName'), miDevice: $('miDevice'),
            miArch: $('miArch'), modelCard: $('modelCard'),
            prompt: $('prompt'), promptMsg: $('promptMsg'), promptCount: $('promptCount'),
            temp: $('temp'), topp: $('topp'), rep: $('rep'), maxTokens: $('maxTokens'),
            maxHint: $('maxHint'),
            tempVal: $('tempVal'), toppVal: $('toppVal'), repVal: $('repVal'),
            maxTokensVal: $('maxTokensVal'),
            genBtn: $('genBtn'), resetBtn: $('resetBtn'),
            abortRow: $('abortRow'), abortBtn: $('abortBtn'),
            emptyState: $('emptyState'), skeleton: $('skeleton'), article: $('article'),
            readBody: $('readBody'), meta: $('meta'),
            actions: $('actions'), copyBtn: $('copyBtn'), clearBtn: $('clearBtn'),
            continueBtn: $('continueBtn'),
            // 历史记录（控制区底部，默认收起）
            historyBox: $('historyBox'), historySum: $('historySum'),
            historyList: $('historyList'), historyNote: $('historyNote'),
            historyFoot: $('historyFoot'), historyClearBtn: $('historyClearBtn'),
            ex1: $('ex1'), ex2: $('ex2'),
            toastHost: $('toastHost'),
        };

        // ===== 模型信息（block_size / max_tokens 默认值的事实来源） =====
        let MODEL = null;          // null = 还没拿到
        // 生成长度滑条的**绝对下限**（HTML 里 min="30"）。取一个较大的下限是为了
        // 保证滑条始终有有效区间 —— 否则输入很长时 max 会掉到 min 以下，滑条失效。
        const maxTokensFloor = 30;
        // 服务端下发的**天花板**（现在是 512 = block_size；同时是后端 pydantic 的校验边界）。
        // 实际滑条上限 = min(天花板, block_size − 输入字数)，见 updateGenCeiling()。
        let maxTokensServerCap = null;

        // ---------------------------------------------------------------
        // 小工具
        // ---------------------------------------------------------------
        function setHidden(node, hide) {
            if (hide) node.setAttribute('hidden', '');
            else node.removeAttribute('hidden');
        }
        // 提示文字：长度/错误共用一个元素，用 dataset.kind 区分谁可以清掉谁
        function showPromptMsg(msg, kind) {
            el.promptMsg.textContent = msg;
            el.promptMsg.dataset.kind = kind || 'length';
            el.promptMsg.className = 'field-msg' + (kind === 'error' ? ' err' : '');
        }
        function clearPromptMsg() {
            el.promptMsg.textContent = '';
            el.promptMsg.dataset.kind = '';
            el.promptMsg.className = 'field-msg';
        }
        function markInvalid(msg) {
            el.prompt.classList.add('invalid');
            showPromptMsg(msg, 'error');
            el.prompt.focus();
        }
        function dropInvalid() {
            el.prompt.classList.remove('invalid');
            if (el.promptMsg.dataset.kind === 'error') clearPromptMsg();
        }

        // ---------------------------------------------------------------
        // toast：替代 alert。不打断操作、可样式化、2.6s 自动消失
        // ---------------------------------------------------------------
        function toast(msg, kind) {
            const t = document.createElement('div');
            t.className = 'toast' + (kind ? ' ' + kind : '');
            t.textContent = msg;
            el.toastHost.appendChild(t);
            setTimeout(() => {
                t.classList.add('hide');
                setTimeout(() => t.remove(), 340);
            }, 2600);
        }

        // ---------------------------------------------------------------
        // 阅读区四态：空 / 加载 / 成功 / 错误（视觉上必须明显不同）
        // ---------------------------------------------------------------
        function showEmpty() {
            setHidden(el.emptyState, false);
            setHidden(el.skeleton, true);
            setHidden(el.article, true);
            el.article.className = 'article';
            el.article.textContent = '';
        }
        function showSkeleton() {
            setHidden(el.emptyState, true);
            setHidden(el.skeleton, false);
            setHidden(el.article, true);
            el.article.className = 'article';
            el.article.textContent = '';
        }

        // ★ 阅读感的最大来源：把生成文本**按段落**渲染成多个 <p>。
        //   模型输出常常只有单换行、没有空行 → 先按空行切，只有 1 段时再按单换行切。
        //   安全：全部走 textContent，绝不把模型输出塞进 innerHTML。
        function toParagraphs(text) {
            const raw = String(text == null ? '' : text);
            let blocks = raw.split(/\n{2,}/).map((s) => s.trim()).filter(Boolean);
            if (blocks.length <= 1) {
                blocks = raw.split(/\n+/).map((s) => s.trim()).filter(Boolean);
            }
            return blocks.length ? blocks : [raw.trim()];
        }
        function fillArticle(text, isError) {
            const blocks = toParagraphs(text);
            el.article.textContent = '';
            for (const b of blocks) {
                const p = document.createElement('p');
                if (isError && b === blocks[0]) p.className = 'err-what';
                if (isError && b === blocks[blocks.length - 1] && blocks.length > 1) p.className = 'err-how';
                p.textContent = b;
                el.article.appendChild(p);
            }
            el.article.className = isError ? 'article error-text' : 'article';
            setHidden(el.emptyState, true);
            setHidden(el.skeleton, true);
            setHidden(el.article, false);
        }
        function showResult(text) { fillArticle(text, false); }
        function showError(msg, how) { fillArticle(msg + '\n\n' + how, true); }

        // ---------------------------------------------------------------
        // 滑条：数值回显 + 已填充比例（自绘轨道靠 --pct 变量）
        // ---------------------------------------------------------------
        const SLIDERS = [
            ['temp', 'tempVal'], ['topp', 'toppVal'],
            ['rep', 'repVal'], ['maxTokens', 'maxTokensVal'],
        ];
        function syncSliders() {
            for (const [input, out] of SLIDERS) {
                const node = el[input];
                el[out].textContent = node.value;
                const min = parseFloat(node.min), max = parseFloat(node.max);
                const pct = max > min ? ((parseFloat(node.value) - min) / (max - min)) * 100 : 0;
                node.style.setProperty('--pct', pct.toFixed(2) + '%');
            }
        }
        for (const [input] of SLIDERS) {
            el[input].addEventListener('input', syncSliders);
        }

        // ---------------------------------------------------------------
        // prompt 校验 + 字数计数（上限来自 /model-info 的 block_size）
        // ---------------------------------------------------------------
        // ★ 注意：后端 generate_ids 会把超长 prompt 静默截断到 block_size（见
        //   model/generation.py:33），不会报错。所以这里必须主动提示，
        //   否则用户会以为"整段输入都被模型看到了"。
        // ★ 单位提醒：block_size 是 **token** 数，这里数的是**字符**数。
        //   中文大致 1 字 ≈ 1 token，所以按"字"给用户提示是够用的近似，
        //   但严格说不是同一单位 —— 别把它当成精确的 token 计数。
        function checkPrompt() {
            const v = el.prompt.value;
            const len = v.length;
            const limit = MODEL && MODEL.block_size ? MODEL.block_size : null;
            updateGenCeiling();          // 上限随输入长度收缩（定义见下方）

            if (!limit) {
                el.promptCount.textContent = len + ' 字';
                el.promptCount.className = 'counter';
            } else {
                const gen = el.maxTokens ? parseInt(el.maxTokens.value, 10) || 0 : 0;
                const total = len + gen;                    // prompt + 计划生成
                const overCombo = total > limit;
                const nearCombo = total > limit * 0.9;

                el.promptCount.textContent = len + ' 字';
                const warn = len > limit || overCombo || nearCombo;
                // 数量本身不变（都是"字数"），只靠颜色区分接近/超限
                el.promptCount.className = 'counter' + (warn ? ' warn' : '');

                if (len > limit) {
                    showPromptMsg('⚠️ 输入已超过模型上下文 ' + limit + ' 字，超出部分会被截断'
                                  + '（模型只看最后 ' + limit + ' 字），前面的内容会被丢弃');
                } else if (overCombo) {
                    // ★ 组合超限：prompt 本身没超，但"prompt + 生成长度"超了。
                    //   后端不是不让你生成 —— generation.py:48-52 会滑动窗口重建，
                    //   生成照常继续，但窗口滑走后 prompt 的开头会被丢掉，
                    //   表现为"写着写着忘了开头写的是什么"。这是体验问题，不是报错。
                    showPromptMsg('⚠️ 输入 ' + len + ' 字 + 计划生成 ' + gen + ' 字 = ' + total
                                  + ' 字，已超过上下文 ' + limit + ' 字。生成到中途时，'
                                  + '开头的输入会滑出模型的记忆窗口（生成不会中断，'
                                  + '但可能逐渐跑题）→ 建议缩短输入或降低生成长度');
                } else if (nearCombo && gen > 0) {
                    showPromptMsg('⚠️ 输入 ' + len + ' 字 + 计划生成 ' + gen + ' 字 = ' + total
                                  + ' 字，接近上下文上限 ' + limit + ' 字');
                } else if (el.promptMsg.dataset.kind === 'length') {
                    clearPromptMsg();
                }
            }
            if (v) dropInvalid();
        }
        el.prompt.addEventListener('input', checkPrompt);
        // max_tokens 变化也要重算组合超限（否则用户拉长生成长度时提示不会更新）
        if (el.maxTokens) el.maxTokens.addEventListener('input', checkPrompt);

        // ---------------------------------------------------------------
        // 生成长度滑条：**按输入长度动态确定上限**
        // ---------------------------------------------------------------
        // 为什么：block_size 限的是注意力窗口（prompt + 生成的总 token 数），
        //   不是生成长度（见 model/generation.py:48-52 的滑动窗口重建）。
        //   所以"还能生成多少"取决于"输入已经占了多少"：
        //      生成长度上限 = block_size - 输入长度
        //   固定写死 300 时，输入 300 字 + 生成 300 字 = 600 > 512 → 中途滑窗丢开头，
        //   表现为"写着写着跑题"。这里让上限随输入收缩，从源头避免踩坑。
        //
        // ⚠️ 单位是**字符**不是 token（前端拿不到精确 token 数）。
        //   中文约 1 字 ≈ 1 token，按字估算是够用的近似；英文会高估 token 数
        //   （偏保守，宁紧不松），别把它当成精确的 token 计数。
        function updateGenCeiling() {
            if (!el.maxTokens) return;
            const limit = MODEL && MODEL.block_size ? MODEL.block_size : null;
            if (!limit) return;                       // 模型信息未到 → 保持原样

            const len = el.prompt ? el.prompt.value.length : 0;
            const room = limit - len;                 // 还能生成多少（字符≈token）
            // ★ 两种模式（2026-09-20 修正，见 commit message）：
            //   A) len <= limit：输入还没占满窗口 → 上限 = limit − len。
            //      这样"输入 + 生成"正好铺满窗口，不需要滑动。
            //   B) len > limit：**输入本身已超窗** → 上限 = 服务端天花板。
            //      此时 prompt 会被 generation.py:33 截到尾部、生成中窗口持续滑动，
            //      内容本来就在"老化淡出"，没有"铺满窗口"可谈 —— 若仍按 room 算会得到
            //      负数、被下限夹到 30，于是「接着写」每次只能续 30 字，功能等于废掉。
            //      ⚠️ 这正是用户此前纠正过的同一类错误（拿一个假约束去压真实需求）。
            const cap = (len > limit)
                ? (maxTokensServerCap || limit)
                : (maxTokensServerCap ? Math.min(maxTokensServerCap, room) : room);
            const max = Math.max(maxTokensFloor, cap);
            el.maxTokens.max = String(max);

            // 当前值超出新上限 → 夹紧（不静默：提示里说明，用户看得到）
            // 超窗模式不夹紧：用户自己选的生成长度是合理的，没理由因为输入变长而没收
            let clamped = false;
            const cur = parseInt(el.maxTokens.value, 10) || 0;
            if (len <= limit && cur > max) {
                el.maxTokens.value = String(max);
                clamped = true;
            }

            if (el.maxHint) {
                const t = parseInt(el.maxTokens.value, 10) || 0;
                // 软提示线：生成窗口内的"远景"会逐渐滑出视野，实践上超过 ~300 字后
                // 容易与开头脱节、句式趋重复。这不是错误，只是质量开始衰减的信号。
                const SOFT = 300;
                const parts = [];
                if (len > limit) {
                    // 超窗：如实说明窗口在滑动，别让用户以为"输入被完整看到了"
                    parts.push('输入 ' + len + ' 字已超过窗口 ' + limit + '：模型只看最后 '
                               + limit + ' 字，生成长度上限 ' + max);
                } else if (len > 0) {
                    parts.push('输入 ' + len + ' 字 → 最多还能生成 ' + max + ' 字'
                               + (t < max ? '（再拉可到 ' + max + '）' : ''));
                } else {
                    parts.push('续写最大 token 数 · 上限随输入自动收缩');
                }
                if (t > SOFT) parts.push('⚠️ 超过 ' + SOFT + ' 字后可能逐渐与开头脱节');
                el.maxHint.textContent = parts.join(' · ');
                el.maxHint.classList.toggle('clamp-note', clamped);
            }
            if (clamped) syncSliders();
        }

        // 一键把生成长度拉到当前允许的最大值（对应"多写点"这个常见意图）
        if (el.maxHint) {
            el.maxHint.addEventListener('click', () => {
                if (!el.maxTokens) return;
                el.maxTokens.value = el.maxTokens.max;
                syncSliders();
                checkPrompt();
            });
        }
        // 注意顺序：先调 updateGenCeiling 夹紧上限，再跑 checkPrompt 出提示
        if (el.maxTokens) {
            el.maxTokens.addEventListener('input', updateGenCeiling);
        }

        // ---------------------------------------------------------------
        // 模型信息：副标题 + 信息条。加载中/失败都有降级文案，绝不留白
        // ---------------------------------------------------------------
        function renderModelInfo(m) {
            MODEL = m;
            if (m.error) { renderModelError(m.error); return; }

            const arch = [
                m.n_layer + ' 层', m.n_embd + ' 维', m.n_head + ' 头',
                (m.ff_type === 'swiglu' ? 'SwiGLU' : m.ff_type.toUpperCase())
                    + (m.ff_hidden ? '(h=' + m.ff_hidden + ')' : ''),
                m.position_encoding === 'rope' ? 'RoPE' : '正弦位置编码',
                'vocab ' + m.vocab_size, 'block ' + m.block_size,
                '参数量 ' + m.params_human,
            ].join(' · ');

            el.miName.textContent = '当前模型：' + m.ckpt_name;
            el.miArch.textContent = arch;
            el.miDevice.textContent = m.device;
            setHidden(el.miDevice, false);
            el.subtitle.textContent = '从零训练的中文续写模型 · ' + m.display;

            // 回退提示：终端早就打印了，但页面上必须也看得见
            if (m.is_fallback) {
                const note = document.createElement('div');
                note.className = 'fallback-note';
                note.textContent = '⚠️ 主权重缺失，已回退到 ' + m.ckpt_name
                    + '（期望的 150M v3_gamma 权重不存在）。当前输出由较小的模型产生，'
                    + '质量与文档里的指标不对应。';
                el.modelCard.appendChild(note);
            }

            // max_tokens 滑条默认值以后端为准（单一事实来源）
            if (m.max_tokens_default) {
                el.maxTokens.value = m.max_tokens_default;
            }
            // 服务端硬上限记下来给 updateGenCeiling() 用；**不要**直接写 el.maxTokens.max，
            // 那是动态值（会随输入收缩），直接写会被下面的 checkPrompt() 立刻覆盖。
            if (m.max_tokens_max) maxTokensServerCap = m.max_tokens_max;
            syncSliders();
            checkPrompt();
        }

        function renderModelError(reason) {
            MODEL = null;
            el.miName.textContent = '模型信息不可用';
            el.miArch.textContent = '未能读取 /model-info：' + reason
                + '（续写功能可能仍可用，但上下文长度上限未知）';
            el.subtitle.textContent = '从零训练的中文续写模型';
        }

        async function loadModelInfo() {
            try {
                const r = await fetch('/model-info');
                if (!r.ok) { renderModelError('HTTP ' + r.status); return; }
                renderModelInfo(await r.json());
            } catch (e) {
                renderModelError('网络错误');
            }
        }

        // ---------------------------------------------------------------
        // 生成（流式 SSE）
        // ---------------------------------------------------------------
        // ★★★ 中止语义（2026-09-21 流式改造后**已变**，别再按旧注释理解）★★★
        //   现在走 POST /generate/stream（SSE）+ 同步生成器：
        //     · 服务端每算出一个 token 就 yield 一帧；
        //     · 客户端 abort() → TCP 断开 → starlette 不再 next() 生成器
        //       → 剩余步数**根本不会执行**。
        //   所以按钮叫「停止生成」：它真的停掉了生成，不是只停前端等待。
        //   ✅ 判据（实测）：中止后 300s 内进程 CPU 增量 **0.3%**（纯空转噪声），
        //      且服务端 `已算` 恰好等于客户端已消费的 token 数 —— 一步都没多算。
        //   ⚠️ 反例警告：**不要**用那条 `[stream] 提前结束` 日志来证明"停了"。
        //      starlette 只是不再拉取，并不保证马上 close() 生成器 —— 实测该日志
        //      2.8s~266s 才打，静默时 300s 都不打。日志是观测手段，不是判据。
        //   ⚠️ 也别信"断流后还会多算几十步"的说法：那是 Python 客户端没真正关掉
        //      socket（Session 未 close / 等 GC）导致的假象 —— 连接还活着，服务端
        //      当然继续算。浏览器 AbortController 会立刻断开。
        //   ⚠️ 但要说准一句：已生成的部分**是有效结果**，不是"没算完的垃圾" ——
        //      模型每步产出的就是那一步的最终答案，后续步骤不会回头改写它。
        //      所以中止后**保留已渲染的半截文字**并正常收尾（进历史、可接着写）。
        //   ⚠️ 若浏览器不支持流式（老 Safari 无 ReadableStream）→ 自动退回旧的
        //      /generate 一次性请求，那条路径的中止**仍然只是"停止等待"**，
        //      此时沿用旧的回退快照逻辑（见 catch 分支）。
        const canAbort = (typeof window.AbortController === 'function');
        const canStream = canAbort && typeof window.ReadableStream === 'function'
            && typeof window.TextDecoder === 'function'
            && typeof window.fetch === 'function';
        let inflight = null;     // 当前请求的 AbortController
        let genLocked = false;   // 是否已有请求进行中（防重入）
        let lastResult = null;   // 上一次成功渲染的结果（「接着写」的起点）
        let lastMetaText = '';
        let lastMetaTitle = '';
        // 流式专属：正文是否已经进入"累积 + 节流"渲染模式
        let streamMode = false;

        // 中止后回到**可用状态**（只用于不支持流式的回退路径：/generate 是一次性
        // 请求，中止时手上没有任何增量文本，只能恢复上一次结果）。
        // 没有上次结果 → 回空态。任何一种都不能卡在骨架行。
        function restoreAfterAbort() {
            if (lastResult !== null) {
                showResult(lastResult);
                el.meta.textContent = lastMetaText;
                el.meta.title = lastMetaTitle;
                el.actions.hidden = false;
            } else {
                showEmpty();
                el.meta.textContent = '';
                el.meta.title = '';
                el.actions.hidden = true;
            }
            el.readBody.scrollTop = 0;
            updateContinueBtn();   // 有上次结果 → 「接着写」可用；没有 → 保持禁用
        }

        // 停止生成（abort 的真实语义见上方说明）
        function abortGenerate() {
            if (!inflight) return;   // 空闲/已结束时点它 = 无操作，别抛异常
            inflight.abort();
        }

        function extractDetail(body, status) {
            // FastAPI 的错误体是 {"detail": ...}；detail 可能是字符串（HTTPException）
            // 也可能是数组（422 校验错误，每项含 loc/msg/type）。
            const d = body && body.detail;
            if (typeof d === 'string') return d;
            if (Array.isArray(d)) {
                return d.map((x) => (x.loc ? x.loc.join('.') + ': ' : '') + (x.msg || JSON.stringify(x))).join('；');
            }
            if (d) return JSON.stringify(d);
            return '服务端返回 HTTP ' + status;
        }

        // 骨架行的高度/宽度是写死的，这里只负责"滚到阅读区开头"
        function scrollToReading() {
            el.readBody.scrollTop = 0;
            el.readBody.scrollIntoView({ block: 'start', behavior: 'smooth' });
        }

        // ---------------------------------------------------------------
        // 流式渲染：累积全文 + 节流重绘 + done 之后才写状态
        // ---------------------------------------------------------------
        // 为什么要节流：toParagraphs()/fillArticle() 是**幂等的全文重渲染**（正确性
        // 没问题，最终 DOM 与一次性 /generate 逐字一致），但它每帧都清空并重建全部
        // <p>。一个 token 一帧（~26ms）意味着每秒重排近 40 次，会破坏用户选区、
        // 让长文闪烁。所以把"重绘"与"到达"解耦：文本立刻累积，重绘按
        // STREAM_RENDER_MS 节流（最后一定补一次，保证收尾完整）。
        // ⚠️ 不用 requestAnimationFrame：后台标签页里 rAF 不触发，切回来才补 ——
        //    用户看到的是"卡住"，而 setTimeout 仍会跑（只是被浏览器降频，可接受）。
        const STREAM_RENDER_MS = 80;
        let streamThrottleTimer = null;
        let streamLastRenderAt = 0;
        let streamAccum = '';
        let streamBuffer = '';           // SSE 分帧缓冲（跨 chunk 的半帧留在这里）
        let streamElapsedMs = null;      // done 帧带来的服务端实测耗时
        let streamNewTokens = null;      // done 帧带来的新生成 token 数
        let streamPrompt = '';           // 本次请求的开头（细栏算"新增字数"要用）
        let streamStartedAt = 0;         // 发请求时刻（#debug 里算首字延迟）
        let streamFirstFrameAt = null;   // 第一帧到达时刻（首字延迟的另一半）
        let streamFrameCount = 0;

        function streamCancelThrottle() {
            if (streamThrottleTimer !== null) {
                clearTimeout(streamThrottleTimer);
                streamThrottleTimer = null;
            }
        }
        function streamRenderNow() {
            streamCancelThrottle();
            streamLastRenderAt = Date.now();
            showResult(streamAccum);
        }
        function streamRenderThrottled() {
            const wait = STREAM_RENDER_MS - (Date.now() - streamLastRenderAt);
            if (wait <= 0) { streamRenderNow(); return; }
            if (streamThrottleTimer !== null) return;   // 已排好一次补绘，不重复排
            streamThrottleTimer = setTimeout(() => {
                streamThrottleTimer = null;
                streamLastRenderAt = Date.now();
                showResult(streamAccum);
            }, wait);
        }

        // 细栏文案（耗时 / 字数 / 参数）。★ 字数口径与原来完全一致：
        // data.text 是**全文**（prompt + 新生成），服务端首帧就带 prompt。
        function buildMetaText(fullText, elapsedMs, newTokens, opts) {
            const o = opts || {};
            const parts = [];
            parts.push(typeof elapsedMs === 'number' && elapsedMs >= 0
                ? '耗时 ' + (elapsedMs / 1000).toFixed(1) + 's' : '耗时未知');
            const fullLen = fullText.length;
            const promptStr = (typeof o.prompt === 'string') ? o.prompt : '';
            // 新生成部分 = 全文去掉开头的 prompt。只用**严格前缀匹配**：
            // clean_text() 会 html.unescape + 过滤控制字符，prompt 里若含
            // "&amp;" 这类实体或控制字符，前缀就对不上 —— 此时宁可不显示，
            // 也不用 indexOf / 去空白之类的猜测去"瞎算"一个错的字数（降级见 else）。
            const prefixOk = promptStr.length > 0 && fullText.startsWith(promptStr);
            const newLen = prefixOk ? fullLen - promptStr.length : null;
            if (newLen !== null) {
                // 新增字数与生成 token 数放一起，方便对照（两者口径不同，故意都保留）
                let add = '新增 ' + newLen + ' 字';
                if (typeof newTokens === 'number') add += '（' + newTokens + ' token）';
                parts.push(add);
                parts.push('全文 ' + fullLen + ' 字');
            } else {
                // 降级：切不出可靠的新增部分 → 只报全文，并写明它包含输入开头。
                // 这一句不能省：否则用户又会把全文当成新增字数（正是原来的 bug）。
                parts.push('全文 ' + fullLen + ' 字（含输入开头，新增字数无法判定）');
                if (typeof newTokens === 'number') parts.push('新增 ' + newTokens + ' token');
            }
            // 中止时如实标注"这次没写完"——否则用户会以为模型只写了这么点
            if (o.interrupted) parts.push('已中断，下文未生成');
            parts.push('t' + o.temp + ' · p' + o.top_p + ' · r' + o.rep
                       + ' · 最多 ' + o.maxTokens + ' token');
            return parts.join(' · ');
        }
        const META_TITLE = 't=温度 · p=top_p · r=重复惩罚 · 「最多 N token」= 本次请求的生成长度上限\n'
            + '字数按字符计（不是 token）：新增 = 本次新生成的文字，全文 = 新增 + 你输入的开头';

        // ★ 收尾：**只有流正常结束 / 用户主动中止**才调用（出错不调用）。
        //   集中在一处是为了守住原来的三条不变量：
        //     · 半截结果**不进历史**（出错路径到不了这里）；
        //     · meta / lastResult / pushHistory / updateContinueBtn 永远成套更新
        //       （不会出现"细栏显示了耗时但「接着写」还锁着"这种中间态）。
        //   用户主动中止是唯一的例外：那次结果**进历史** —— 见 abortGenerate 的说明，
        //   已生成的部分是有效文本，用户点停止往往正是想留下它。
        function finalizeStream(interrupted) {
            streamCancelThrottle();
            const fullText = streamAccum;
            const params = {
                temperature: parseFloat(el.temp.value),
                top_p: parseFloat(el.topp.value),
                repetition_penalty: parseFloat(el.rep.value),
                max_tokens: parseInt(el.maxTokens.value, 10),
            };
            const meta = buildMetaText(fullText, streamElapsedMs, streamNewTokens, {
                prompt: streamPrompt,
                interrupted: !!interrupted,
                temp: params.temperature, top_p: params.top_p,
                rep: params.repetition_penalty, maxTokens: params.max_tokens,
            });
            el.meta.textContent = meta;
            el.meta.title = META_TITLE;
            lastResult = fullText;
            lastMetaText = meta;
            lastMetaTitle = el.meta.title;
            el.actions.hidden = false;
            pushHistory(streamPrompt, fullText, params);
            updateContinueBtn();
            // #debug 探针：加这个 hash 打开后，控制台能看到"首字延迟 / 帧数 / 总耗时"，
            // 不用开 devtools 网络面板就能确认"文字真的是一帧一帧来的"。
            // 平时（无 #debug）零开销、不打印任何东西。
            if (location.hash === '#debug' && window.console) {
                console.log('[stream] 首字延迟 %dms · 帧数 %d · 全文 %d 字 · 服务端耗时 %sms · 中断=%s',
                            streamFirstFrameAt === null ? -1 : (streamFirstFrameAt - streamStartedAt),
                            streamFrameCount, fullText.length, streamElapsedMs, !!interrupted);
            }
        }

        // 打开流式渲染：调用点**必须**是成功 2xx 之后（否则半截的骨架会被当成结果）
        function streamBegin(prompt) {
            streamMode = true;
            streamStartedAt = Date.now();
            streamAccum = '';
            streamPrompt = prompt;
            streamElapsedMs = null;
            streamNewTokens = null;
            streamFrameCount = 0;
            streamFirstFrameAt = null;
            streamLastRenderAt = 0;
            streamCancelThrottle();
        }

        // 逐帧解析。返回 true = 收到 done 帧（流正常走完）
        function streamFeedText(text) {
            streamBuffer += text;
            let done = false;
            let idx;
            while ((idx = streamBuffer.indexOf('\n\n')) >= 0) {
                const block = streamBuffer.slice(0, idx);
                streamBuffer = streamBuffer.slice(idx + 2);
                if (streamHandleFrame(block)) done = true;
            }
            return done;
        }

        // 一帧 SSE = 若干 `event:`/`data:` 行 + 一个空行（已在 streamFeedText 剥掉）
        function streamHandleFrame(block) {
            let event = '';
            const dataLines = [];
            for (const line of block.split('\n')) {
                if (line.startsWith('event:')) event = line.slice(6).trim();
                else if (line.startsWith('data:')) dataLines.push(line.slice(5).replace(/^ /, ''));
                // 其余行（注释/未知字段）按 SSE 规范忽略
            }
            if (!dataLines.length) return false;
            let payload = null;
            try { payload = JSON.parse(dataLines.join('\n')); } catch (e) { payload = null; }
            if (!payload) return false;

            if (event === 'error') {
                throw new Error(typeof payload.message === 'string' && payload.message
                    ? payload.message : '服务端流式生成出错');
            }
            if (event === 'done') {
                if (typeof payload.elapsed_ms === 'number') streamElapsedMs = payload.elapsed_ms;
                if (typeof payload.new_tokens === 'number') streamNewTokens = payload.new_tokens;
                return true;
            }
            if (typeof payload.delta === 'string' && payload.delta) {
                if (streamFrameCount === 0) streamFirstFrameAt = Date.now();
                streamFrameCount += 1;
                streamAccum += payload.delta;
                streamRenderThrottled();
            }
            return false;
        }

        // 读一条流式响应，返回 done 帧是否到达
        async function consumeStream(response) {
            streamBuffer = '';
            const reader = response.body.getReader();
            // stream:true 让 TextDecoder 自己缓冲"跨 chunk 被切开的半个多字节字符"
            // —— 与服务端的增量解码是同一个问题的两层（网络层 / token 层）
            const decoder = new TextDecoder('utf-8');
            let done = false;
            for (;;) {
                const r = await reader.read();
                if (r.done) break;
                if (r.value && r.value.length) {
                    done = streamFeedText(decoder.decode(r.value, { stream: true })) || done;
                }
            }
            // 收尾：把解码器里可能残留的半个字符吐出来（流正常结束时不会有残字，
            // 但帧缓冲里可能还剩一个没被 \n\n 收尾的块 —— 一并处理，不丢内容）
            streamFeedText(decoder.decode());
            return done;
        }

        async function generate() {
            // 防重入（二道保险）：按钮已被 disabled，这里再挡一次快捷键 / 程序化调用。
            // 中止后 genLocked 会在 finally 里复位，所以「停止生成」完可以立刻重新生成。
            if (genLocked) return;

            const prompt = el.prompt.value.trim();
            // 空输入：内联提示 + 红边，不再用 alert 打断
            if (!prompt) { markInvalid('请先输入开头文字'); return; }
            dropInvalid();

            const temp = parseFloat(el.temp.value);
            const top_p = parseFloat(el.topp.value);
            const rep = parseFloat(el.rep.value);
            const maxTokens = parseInt(el.maxTokens.value, 10);

            // 本次请求的 controller 存进**局部变量**：finally / catch 只认这一个，
            // 绝不会误清或误回退"用户紧接着发起的新请求"的状态。
            const controller = canAbort ? new AbortController() : null;
            inflight = controller;
            const myRequest = controller;

            // 快照"上次已渲染的结果"：**只给回退路径**用（不支持流式的浏览器走
            // /generate 一次性请求，中止时手上没有增量文本，只能恢复上一次结果）。
            // 流式路径不靠它 —— 流式中止保留的是本次已生成的半截文字。
            // 先快照再 showSkeleton() —— showSkeleton() 会清空 el.article。
            if (!el.article.hidden) {
                lastResult = el.article.textContent;
                lastMetaText = el.meta.textContent;
                lastMetaTitle = el.meta.title;
            }

            genLocked = true;
            // 防连点：禁用按钮 + 状态切换
            el.genBtn.disabled = true;
            el.genBtn.textContent = '正在续写…';
            el.resetBtn.disabled = true;
            updateContinueBtn();   // 「接着写」同样锁住（结果条此刻也会被隐藏，双保险）
            // 「停止生成」只在请求进行中出现（不支持 AbortController 的浏览器 → 永不显示）
            el.abortBtn.disabled = false;
            setHidden(el.abortRow, !canAbort);
            el.meta.textContent = '';
            el.actions.hidden = true;
            showSkeleton();
            scrollToReading();   // 不阻塞用户滚动：只是文本，没有遮罩

            streamMode = false;
            try {
                // ★ 流式优先：POST /generate/stream（SSE，逐 token 到达）。
                //   不支持流式的浏览器退回旧的 /generate（字段/行为一字未改）。
                const useStream = canStream;
                const response = await fetch(useStream ? '/generate/stream' : '/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    // signal 必须传进去，否则 abort() 无法中断这次等待/这条流
                    signal: myRequest ? myRequest.signal : undefined,
                    body: JSON.stringify({
                        prompt: prompt,
                        max_tokens: maxTokens,
                        temperature: temp,
                        top_p: top_p,
                        repetition_penalty: rep,
                    }),
                });

                // ★ 先判 response.ok：FastAPI 的 422/500 都是 JSON 体（非 2xx），
                //   必须走旧的 response.text() + JSON.parse + extractDetail() 路径。
                //   ⚠️ 2xx 才进流 —— 否则会把错误 JSON 当正文渲染出来。
                if (!response.ok) {
                    // 错误体很小，一次性读没问题（流式响应出错时这里也不会被调用）
                    const raw = await response.text();
                    let data = null;
                    try { data = JSON.parse(raw); } catch (e) { /* 非 JSON（如 500 纯文本） */ }
                    const detail = data ? extractDetail(data, response.status) : (raw || '未知错误');
                    showError('生成失败（HTTP ' + response.status + '）：' + detail,
                              '请求没有成功，模型状态未被修改。可以检查后端进程是否还在运行，'
                              + '或把参数调回合理范围后重试。');
                    toast('生成失败，详情见阅读区', 'err');
                    return;
                }

                if (useStream) {
                    streamBegin(prompt);
                    const gotDone = await consumeStream(response);
                    if (!gotDone) {
                        // 流结束但没有 done 帧 = 连接被中途掐断（服务端崩/代理超时）。
                        // 不装作成功，但也不丢掉已经收到的正文 —— 如实标注。
                        streamRenderNow();
                        if (!streamAccum) {
                            showError('流式中断：没有收到任何内容',
                                      '连接在收到第一段文字之前就结束了。请确认后端仍在运行后重试。');
                            toast('流式中断', 'err');
                            return;
                        }
                        finalizeStream(true);
                        toast('流已中断，以上为中途中止前已生成的内容', 'err');
                        return;
                    }
                    // 收尾补绘一次：节流可能还压着最后一帧没画
                    streamRenderNow();
                    finalizeStream(false);
                    scrollToReading();
                    return;
                }

                // ---------------- 回退路径：旧的 /generate ----------------
                const raw = await response.text();
                let data = null;
                try { data = JSON.parse(raw); } catch (e) { /* 非 JSON */ }
                if (!data || typeof data.text !== 'string') {
                    showError('生成失败：服务端返回的不是预期的 JSON',
                              '请检查后端版本是否与前端一致（异常时 /generate 会返回错误详情）。');
                    toast('生成失败，详情见阅读区', 'err');
                    return;
                }

                showResult(data.text);
                lastResult = data.text;   // 供"中止回退"用：记录刚渲染成功的结果
                const meta = buildMetaText(data.text, data.elapsed_ms, data.new_tokens, {
                    prompt: (typeof data.prompt === 'string') ? data.prompt : '',
                    interrupted: false,
                    temp: temp, top_p: top_p, rep: rep, maxTokens: maxTokens,
                });
                el.meta.textContent = meta;
                lastMetaText = meta;      // 与 lastResult 成套保存，供中止回退
                // 悬停解释缩写与字数口径（只设 title 属性，不碰任何视觉样式）
                el.meta.title = META_TITLE;
                lastMetaTitle = el.meta.title;
                el.actions.hidden = false;

                // 历史记录：只有"完整成功"的这一次生成才入库。
                pushHistory(prompt, data.text, {
                    temperature: temp, top_p: top_p,
                    repetition_penalty: rep, max_tokens: maxTokens,
                });
                updateContinueBtn();   // 有完整结果了 → 「接着写」可用

                // 生成完成后把视线带到结果开头（长文时用户不用自己找）
                scrollToReading();
            } catch (e) {
                // ★ 中止必须与真正的网络失败分开处理：
                //   AbortError 是**用户主动操作**（点了「停止生成」），不是故障。
                if (e && e.name === 'AbortError') {
                    if (streamMode) {
                        // 流式：**保留已生成的部分**并正常收尾。
                        //   用户点停止是想留下已写的内容，不是想让它消失。
                        //   若一帧都还没到（prefill 阶段就停），下面不会有内容可留，
                        //   退回骨架 → 空态/上一次结果，别留一个空白阅读区。
                        streamRenderNow();
                        if (streamAccum) {
                            finalizeStream(true);
                            toast('已停止生成（已保留写出的部分，可接着写）');
                        } else {
                            restoreAfterAbort();
                            toast('已停止生成（还没开始输出文字）');
                        }
                    } else {
                        // 回退路径（/generate）：中止只是停止前端等待，服务端那次推理
                        // 照旧算完，所以手上什么都没有 → 恢复上一次结果。
                        restoreAfterAbort();
                        toast('已停止等待（服务端可能仍在完成本次请求）');
                    }
                } else {
                    // 只有真正的网络层失败 / 流中途出错才到这里（4xx/5xx 已在上面处理）。
                    // ⚠️ 流式下若已经渲染了正文 → 不覆盖成错误态（那些文字是真的），
                    //    改走"保留 + 标注中断"。
                    if (streamMode && streamAccum) {
                        streamRenderNow();
                        finalizeStream(true);
                        toast('生成中断：' + (e && e.message ? e.message : '流读取失败'), 'err');
                    } else {
                        showError('请求失败：' + (e && e.message ? e.message : '无法连接服务端'),
                                  '请确认后端进程还在运行（python app/server.py），然后重试。');
                        el.meta.textContent = '';
                        toast('网络错误', 'err');
                    }
                }
            } finally {
                // 状态复位（中止 / 成功 / 失败都要走这里 → 按钮总会恢复可用）。
                // ⚠️ 只在"本次请求"仍是当前 inflight 时才清空：用户中止后立刻重新生成时，
                //    inflight 已被新请求的 controller 覆盖，这里不能把它置空、
                //    更不能覆盖新请求的 UI 状态（genLocked / 按钮文案 / 停止按钮可见性）。
                if (inflight === myRequest) {
                    inflight = null;
                    genLocked = false;
                    setHidden(el.abortRow, true);
                    el.abortBtn.disabled = true;
                    el.genBtn.disabled = false;
                    el.genBtn.textContent = '开始续写';
                    el.resetBtn.disabled = false;
                    // 中止后若还留着上次结果 → 「接着写」重新可用（有起点才能续）
                    updateContinueBtn();
                }
            }
        }

        // ---------------------------------------------------------------
        // 连续续写（「接着写」）
        // ---------------------------------------------------------------
        // 用户意图：写完一段，最自然的动作是"接着往下写"，而不是只能复制或清空。
        // 做法：把**当前全文**接回输入框，立刻再生成一段 → 下一轮的 prompt =
        //       上一轮的全文，于是可以一段接一段地滚下去。
        //
        // ★★★ 这条链子**依赖**后端的一个"截断"行为，改之前务必读懂 ★★★
        //   model/generation.py 的 stream_ids() / generate_ids() 里
        //   `prompt_ids = list(prompt_ids[-block:])`
        //   —— 超过 block_size（本机 512）的 prompt 会被**静默砍掉头部、只留尾部**。
        //   连续续写两三轮后 prompt 必然远超 512，于是每次真正进模型的只有最近 512 个
        //   token。这**正好是我们要的**：接着往下写靠的是上下文的尾部（语气的惯性、
        //   刚刚发生的事），尾窗足够；反过来，把尾巴丢掉才会写得驴唇不对马嘴。
        //   ⚠️ 所以这**不是丢数据的 bug，不要去"修复"它**（比如改成拒绝超长输入、
        //      或者偷偷把输入截短再发）—— 那等于把连续续写直接做废。
        //   必须分清两件事：**输入框里是全文**（用户看得见的完整文本，也是历史记录里
        //   存的那一份），被截断的只是"模型这一轮能看到的部分"。
        function continueWriting() {
            if (genLocked) return;                       // 防重入：与「开始续写」同一把锁
            if (lastResult === null) {                   // 没有上一轮结果 = 没有续写起点
                toast('还没有可续写的正文');
                return;
            }

            el.prompt.value = lastResult;                // ① 全文接回输入框（可见内容同步更新）
            checkPrompt();                               // ② 计数 / 上限 / 超限提示重算

            // ★ 超长提示怎么处理（本次最需要判断力的地方）：
            //   checkPrompt() 会算出"输入 780 字 > 上下文 512 字"，技术上完全正确；
            //   但对"点了接着写"的用户来说，"超出部分会被截断/前面的内容会被丢弃"
            //   读起来像**我们把他刚写的字弄丢了**。所以这里在它之后覆写成一句**如实**的
            //   说明：字一个都没丢（此刻就在输入框里、上一轮还在历史里），被限制的只是
            //   "模型这一轮能看到多远" —— 连带说明这一次的结果会从那段尾部开始。
            //   ⚠️ 绝不为了让提示消失而偷偷缩短输入 —— 那是在骗用户。
            const limit = (MODEL && MODEL.block_size) ? MODEL.block_size : null;
            const len = el.prompt.value.length;
            if (limit && len > limit) {
                // checkPrompt() 刚刚按既有规则收缩过生成长度时（maxHint 上的 clamp-note），
                // 必须一并说明 —— 否则用户会奇怪"为什么这次只写了这么一小段"。
                const clamped = !!(el.maxHint && el.maxHint.classList.contains('clamp-note'));
                const budgetNote = clamped
                    ? '另外：生成长度已按既有规则收缩到 ' + el.maxTokens.value + ' token'
                      + '（开头越长、剩下的生成预算越少）；想一次写长一点，可以先把输入删短一些。'
                    : '';
                showPromptMsg('⚠️ 已把全文 ' + len + ' 字接回输入框（连续续写就是这样工作的）。'
                              + '模型这一轮只看**最后 ' + limit + ' token**（中文大致 1 字 ≈ 1 token）：'
                              + '这是上下文窗口的硬限制，不是把字弄丢了 —— 全文此刻就在输入框里，'
                              + '但这一次的结果会从那段尾部开始，前面约 ' + Math.max(0, len - limit)
                              + ' 字不会出现在结果里（上一轮的完整正文仍在历史记录里）。'
                              + budgetNote, 'length');
            } else if (limit) {
                showPromptMsg('已把全文 ' + len + ' 字接回输入框，这次接着往下写。', 'length');
            }

            // 光标与视口都落到末尾：用户一眼就能看到"续写的起点"是全文的结尾
            el.prompt.setSelectionRange(len, len);
            el.prompt.scrollTop = el.prompt.scrollHeight;
            generate();                                  // ③ 立即生成下一段
        }

        // 「接着写」的可用性：必须有**上一轮完整结果**，且当前没有请求在飞。
        // （首次进入 / 结果被清空 / 生成进行中 → disabled；中止后若仍有上次结果 → 可用）
        function updateContinueBtn() {
            if (!el.continueBtn) return;
            el.continueBtn.disabled = genLocked || lastResult === null;
        }

        // ---------------------------------------------------------------
        // 复制：三级降级链
        //   ⚠️ --host 默认 0.0.0.0，局域网用 http://192.168.x.x:8000 访问时
        //   **不是 secure context** → navigator.clipboard 必然 reject 或不存在。
        //   所以必须有 execCommand 兜底，光提示用户"手动复制"是不够的。
        // ---------------------------------------------------------------
        function selectResultText() {
            const node = el.article;
            const range = document.createRange();
            range.selectNodeContents(node);
            const sel = window.getSelection();
            sel.removeAllRanges();
            sel.addRange(range);
        }

        function legacyCopy(text) {
            // execCommand 已 deprecated，但 HTTP 页面里这是唯一可行的兜底，别无选择
            try {
                const ta = document.createElement('textarea');
                ta.value = text;
                ta.setAttribute('readonly', '');
                ta.style.cssText = 'position:fixed;top:0;left:-9999px;opacity:0;';
                document.body.appendChild(ta);
                ta.select();
                ta.setSelectionRange(0, ta.value.length);
                const ok = document.execCommand('copy');
                document.body.removeChild(ta);
                return ok;
            } catch (e) {
                return false;
            }
        }

        // 从渲染后的段落里取回纯文本：段间补空行，复制出来仍是分段的小说
        function currentArticleText() {
            const ps = el.article.querySelectorAll('p');
            if (!ps.length) return el.article.textContent.trim();
            return Array.prototype.map.call(ps, (p) => p.textContent.trim()).join('\n\n');
        }

        async function copyResult() {
            if (el.article.hidden) { toast('还没有内容可复制'); return; }
            const text = currentArticleText();
            if (!text) { toast('还没有内容可复制'); return; }
            // 1) 现代 API（仅 secure context 可用）
            if (navigator.clipboard && navigator.clipboard.writeText) {
                try {
                    await navigator.clipboard.writeText(text);
                    toast('已复制', 'ok');
                    return;
                } catch (e) { /* 落到 2) */ }
            }
            // 2) execCommand 兜底
            if (legacyCopy(text)) { toast('已复制', 'ok'); return; }
            // 3) 彻底失败：替用户选中，并说清楚下一步
            selectResultText();
            toast('复制失败，已为你选中，请按 Ctrl+C', 'err');
        }

        // ---------------------------------------------------------------
        // 清空 / 恢复默认 / 快捷键
        // ---------------------------------------------------------------
        function clearResult() {
            showEmpty();
            el.meta.textContent = '';
            el.actions.hidden = true;
            // ★ 清空 = 用户明确丢弃这份结果 → 把"接着写"的起点一起清掉，
            //   也把回退路径（不支持流式的浏览器）要用的快照清掉。
            //   否则清空后再发一次请求、中途点「停止生成」：流式路径会保留半截正文、
            //   回退路径的 restoreAfterAbort() 会把刚被清掉的旧正文"复活"出来。
            //   （历史遗留行为，顺手修掉。）
            // ★ 清空**只影响阅读区**：历史记录是独立的一份，照旧保留（见报告里的取舍）。
            lastResult = null;
            lastMetaText = '';
            lastMetaTitle = '';
            updateContinueBtn();
            el.readBody.scrollTop = 0;
        }

        function resetAll() {
            el.temp.value = 0.8;
            el.topp.value = 0.9;
            el.rep.value = 1.15;
            // 生成长度的默认值同样以后端为准
            el.maxTokens.value = (MODEL && MODEL.max_tokens_default) ? MODEL.max_tokens_default : 100;
            syncSliders();
            el.prompt.value = '';
            el.prompt.classList.remove('invalid');
            clearPromptMsg();
            checkPrompt();
            el.prompt.focus();
            toast('已恢复默认参数', 'ok');
        }

        // 示例开头：点击/回车填入 textarea（不自动提交，用户还能改）
        function useExample(node) {
            el.prompt.value = node.textContent.trim();
            checkPrompt();
            el.prompt.focus();
            el.prompt.setSelectionRange(el.prompt.value.length, el.prompt.value.length);
        }
        for (const k of ['ex1', 'ex2']) {
            el[k].addEventListener('click', () => useExample(el[k]));
            el[k].addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); useExample(el[k]); }
            });
        }

        el.prompt.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') { e.preventDefault(); generate(); }
        });
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') { clearResult(); el.prompt.focus(); }
        });

        el.genBtn.addEventListener('click', generate);
        el.resetBtn.addEventListener('click', resetAll);
        el.abortBtn.addEventListener('click', abortGenerate);
        el.copyBtn.addEventListener('click', copyResult);
        el.clearBtn.addEventListener('click', clearResult);
        el.continueBtn.addEventListener('click', continueWriting);
        el.historyClearBtn.addEventListener('click', clearHistory);

        // ---------------------------------------------------------------
        // 历史记录（localStorage）
        // ---------------------------------------------------------------
        // 目的：刷新页面不再"几百字全没了"。
        // 契约：只碰浏览器本地存储，**不改后端、不改 /generate 的字段**。
        //
        // ★ 可用性只探测**一次**（不是每次操作都 try）：无痕模式、被禁用的
        //   localStorage、企业策略、配额为 0 都会让访问直接抛异常。探测失败 →
        //   整体降级（如实说明 + 只在当前页面内保留内存列表），
        //   绝不每次点击都抛一次异常。
        const HISTORY_KEY = 'minigpt_history_v1';   // ★ 带版本号：以后改结构就换 key
        const HISTORY_MAX = 20;                     // 最多 20 条，超了丢最旧的
        const HISTORY_TITLE_CHARS = 14;             // 标题取开头十几个字

        function detectStorage() {
            try {
                const probe = '__minigpt_probe__';
                window.localStorage.setItem(probe, '1');
                window.localStorage.removeItem(probe);
                return window.localStorage;
            } catch (e) {
                return null;          // 不可用。调用方一律先判 null，别再抛
            }
        }
        let storage = detectStorage();   // null = 本浏览器不可用
        let history = [];                // 内存里的那一份（storage 挂了也还能用）
        let historyWarned = false;       // 写失败只提示一次，不刷屏

        // 单条记录的形状校验：旧版本格式 / 被手改过 → 直接丢掉这一条
        function isHistoryEntry(x) {
            return !!x && typeof x === 'object'
                && typeof x.ts === 'number' && isFinite(x.ts)
                && typeof x.prompt === 'string' && typeof x.result === 'string';
        }

        // 读取：JSON.parse 必须在 try 里 —— 手改过的脏数据、旧版本格式、读取异常，
        // 一律当"空历史"，**绝不把异常抛给首页初始化**（否则整个页面会白屏）。
        function loadHistory() {
            if (!storage) return [];
            try {
                const raw = storage.getItem(HISTORY_KEY);
                if (!raw) return [];
                const arr = JSON.parse(raw);
                if (!Array.isArray(arr)) return [];
                return arr.filter(isHistoryEntry).slice(0, HISTORY_MAX);
            } catch (e) {
                return [];
            }
        }

        // 写入：setItem 是最可能抛的地方（配额满 / 隐私模式中途失效）。
        // 失败不抛、也不静默：第一次给一句如实提示，之后整体降级为"仅本次会话"。
        function saveHistory() {
            if (!storage) return;
            try {
                storage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, HISTORY_MAX)));
            } catch (e) {
                storage = null;       // 一次性降级：之后不再尝试写/读
                if (!historyWarned) {
                    historyWarned = true;
                    toast('本地存储不可用，历史只保留在当前页面（刷新会丢）', 'err');
                }
            }
        }

        function pad2(n) { return String(n).padStart(2, '0'); }

        // 时间戳 → 紧凑显示：今天的只写 HH:mm，更早的写 MM-DD HH:mm
        function fmtTime(ts) {
            const d = new Date(ts);
            if (isNaN(d.getTime())) return '';
            const now = new Date();
            const sameDay = d.getFullYear() === now.getFullYear()
                && d.getMonth() === now.getMonth() && d.getDate() === now.getDate();
            const hm = pad2(d.getHours()) + ':' + pad2(d.getMinutes());
            return sameDay ? hm : pad2(d.getMonth() + 1) + '-' + pad2(d.getDate()) + ' ' + hm;
        }

        // 标题 = 开头的 prompt 压成一行取前十几个字（prompt 为空时退用 result）
        function historyTitle(e) {
            const raw = (e.prompt || e.result || '').replace(/\s+/g, ' ').trim();
            if (!raw) return '（空）';
            return raw.length > HISTORY_TITLE_CHARS
                ? raw.slice(0, HISTORY_TITLE_CHARS) + '…' : raw;
        }

        // 渲染：★ 用户/模型文本**一律走 textContent**，绝不用 innerHTML
        function renderHistory() {
            const n = history.length;
            // 收起时那一行的计数：每次数据变化都要刷新
            el.historySum.textContent = storage
                ? '历史记录 (' + n + ')'
                : '历史记录 (' + n + ') · 存储不可用';

            if (!storage) {
                el.historyNote.textContent = '⚠️ 本浏览器不可用本地存储（隐私模式 / 被禁用 / 配额满）：'
                    + '历史只保留在当前页面，刷新后会丢。'
                    + (n ? '下面这些条目仍可点开恢复。' : '');
            } else if (n === 0) {
                el.historyNote.textContent = '生成成功后会自动存一条（最多 ' + HISTORY_MAX
                    + ' 条），只存在这台浏览器里。';
            } else {
                el.historyNote.textContent = '点任意一条可恢复它的正文、开头与参数。最多 '
                    + HISTORY_MAX + ' 条，只存在这台浏览器里。';
            }

            el.historyList.textContent = '';      // 清空子节点（不是 innerHTML）
            history.forEach((e, i) => {
                const b = document.createElement('button');
                b.type = 'button';
                b.className = 'history-item';
                b.title = '恢复这条记录（' + fmtTime(e.ts) + '）';

                const t = document.createElement('span');
                t.className = 'history-item-title';
                t.textContent = historyTitle(e);   // ★ 模型输出永不进 innerHTML

                const d = document.createElement('span');
                d.className = 'history-item-time';
                d.textContent = fmtTime(e.ts);

                b.appendChild(t);
                b.appendChild(d);
                b.addEventListener('click', () => restoreHistory(i));
                el.historyList.appendChild(b);
            });
            setHidden(el.historyFoot, n === 0);    // 没条目就不显示「清除历史」
        }

        // 入库：只在**完整成功**的一次生成后调用（中止 / 报错都到不了这里）
        function pushHistory(prompt, result, params) {
            // 同一份 prompt + result 已存在 → 只更新时间并提到最前，不堆重复条目
            const dup = history.findIndex((x) => x.prompt === prompt && x.result === result);
            if (dup >= 0) history.splice(dup, 1);
            history.unshift({ ts: Date.now(), prompt: prompt, result: result, params: params });
            if (history.length > HISTORY_MAX) history.length = HISTORY_MAX;   // 丢最旧的
            saveHistory();
            renderHistory();
        }

        // 恢复一条：prompt + result + 四个参数**一起**恢复。
        // ★ 只恢复 prompt 是不够的 —— 否则用户点「接着写」时用的是当前滑条上的
        //   新参数，而这条正文是旧参数生成的，会对不上、很困惑。
        function restoreHistory(idx) {
            const e = history[idx];
            if (!e) return;
            if (genLocked) { toast('正在续写，等这次结束再恢复'); return; }

            el.prompt.value = e.prompt;            // ① 开头
            // ★ 顺序有讲究：先把开头写进去并**立刻**重算上限，再恢复参数。
            //   否则上一次长输入把 max_tokens 滑条的 max 收缩到 30 之后，这里拿
            //   旧上限去校验"要恢复的 100"会判成越界 → 参数被静默丢掉。
            checkPrompt();
            const p = (e.params && typeof e.params === 'object') ? e.params : {};
            // 数值合法且在当前滑条量程内才写入（旧格式 / 被手改过 → 保留当前值，不写 NaN）
            const setSlider = (node, v) => {
                if (!node || typeof v !== 'number' || !isFinite(v)) return;
                const min = parseFloat(node.min), max = parseFloat(node.max);
                if (v < min || v > max) return;
                node.value = String(v);
            };
            setSlider(el.temp, p.temperature);     // ② 四个参数
            setSlider(el.topp, p.top_p);
            setSlider(el.rep, p.repetition_penalty);
            // ⚠️ 生成长度放最后：checkPrompt() → updateGenCeiling() 会按**恢复后的开头长度**
            //    再夹紧一次 —— 恢复出来的 max_tokens 仍可能被合理地压低（不是丢数据）
            setSlider(el.maxTokens, p.max_tokens);
            syncSliders();
            checkPrompt();

            showResult(e.result);                  // ③ 正文（必须一起恢复，不能只剩开头）
            lastResult = e.result;
            // 历史里没存耗时 / token 数 → 这里**如实说不显示**，不编一个数字出来。
            // 参数也可能缺（旧版本记录）→ 缺就写"参数未知"，不打印 undefined。
            const pv = (v) => (typeof v === 'number' && isFinite(v)) ? String(v) : null;
            const pParts = [];
            if (pv(p.temperature)) pParts.push('t' + pv(p.temperature));
            if (pv(p.top_p)) pParts.push('p' + pv(p.top_p));
            if (pv(p.repetition_penalty)) pParts.push('r' + pv(p.repetition_penalty));
            if (pv(p.max_tokens)) pParts.push('最多 ' + pv(p.max_tokens) + ' token');
            el.meta.textContent = '历史记录 · ' + fmtTime(e.ts) + ' · '
                + (pParts.length ? '参数 ' + pParts.join(' · ') : '参数未知（这条来自旧版本的记录）');
            el.meta.title = '这是从本地历史恢复的正文。存的时候只留了正文 / 开头 / 参数，'
                + '没有存耗时与 token 数，所以细栏里不显示它们。';
            lastMetaText = el.meta.textContent;
            lastMetaTitle = el.meta.title;
            el.actions.hidden = false;
            updateContinueBtn();
            el.readBody.scrollTop = 0;
            toast('已恢复这条记录', 'ok');
        }

        // 清除历史：清的是**浏览器本地存储**，与阅读区的「清空」互不影响
        function clearHistory() {
            history = [];
            if (storage) {
                try { storage.removeItem(HISTORY_KEY); } catch (e) { /* 删不掉也不崩 */ }
            }
            renderHistory();
            toast('已清除历史记录', 'ok');
        }

        // 初始化：读一次历史（脏数据 → 空历史）。storage 不可用时 UI 照常初始化，
        // 只是文案改成如实说明 —— 不藏起入口，用户至少知道"为什么这里没有历史"。
        history = loadHistory();
        renderHistory();

        // ===== 初始化 =====
        syncSliders();
        checkPrompt();
        updateContinueBtn();   // 首次进入：没有结果 → 「接着写」禁用
        loadModelInfo();
    })();
    