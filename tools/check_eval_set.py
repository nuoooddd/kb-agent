# -*- coding: utf-8 -*-
r"""
check_eval_set.py —— 评估集质量自检

评估集里每条题都靠 must_contain 做字符串匹配判分，所以「选的词是不是只落在
正确的那一个 chunk 里」直接决定指标可不可信：选到满篇通用的词，题没答对也会
被判成命中（假阳性）；选到不在同一 chunk 里的词，永远判不中（假阴性）。

本脚本读 index.json 与 data/eval_set.json，对每条题模拟 eval.py 的判定逻辑，
给出四类结论：
    [OK]      只有一块同时含齐全部关键词  -> 判分正确
    [OK·重叠] 多块命中，但那些块共享同一段长文本（切分时 overlap 复制）
              -> 答案在每块里都完整，命中哪块都对，可保留
    [多处]    多块命中，且分属不同段落内容
              -> 假阳性风险，需要换更专用的词
    [零]      没有任何一块含齐全部关键词
              -> 永远判不中，该条无效

用法：
    python tools/check_eval_set.py
"""
import difflib
import json
import os
import sys
import collections

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))

# 脚本在 tools/ 下 -> 项目根是上一级
ROOT = os.path.dirname(BASE)
if not os.path.exists(os.path.join(ROOT, "index.json")):
    ROOT = os.path.join(BASE, "kb-agent")

INDEX_PATH = os.path.join(ROOT, "index.json")
EVAL_PATH = os.path.join(ROOT, "data", "eval_set.json")


def load_json(path, label):
    if not os.path.exists(path):
        print("[错误] 找不到%s：%s" % (label, path))
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print("[错误] %s 不是合法的 JSON" % label)
        print("       位置：第 %d 行，第 %d 列" % (e.lineno, e.colno))
        print("       原因：%s" % e.msg)
        msg = e.msg
        if "Expecting ',' delimiter" in msg:
            print("       -> 最常见：上一条的 } 后面漏了逗号")
        elif "Extra data" in msg:
            print("       -> 最常见：最后一条的 } 后面多了逗号，或者文件里粘了两段")
        elif "Expecting property name" in msg:
            print("       -> 最常见：多了一个逗号，比如 ] 前面留了 ,")
        elif "Expecting value" in msg:
            print("       -> 最常见：值写成了中文引号，或者少了引号")
        return None


def count_blocks(records, word):
    return sum(1 for r in records if word in r.get("text", ""))


def answer_window(text, keys, pad=50):
    """把答案那句话连前后 pad 个字截出来，用于比对不同块是不是同一句话。"""
    pos = []
    for k in keys:
        p = text.find(k)
        if p < 0:
            return ""
        pos.append((p, p + len(k)))
    lo = max(0, min(p for p, _ in pos) - pad)
    hi = max(e for _, e in pos) + pad
    return text[lo:hi]


def is_overlap_repeat(texts, keys, min_len=60):
    """多块之间是否共享一段很长的、且包含全部关键词的原文。

    chunk 切分时相邻块会重叠（本项目 overlap=100 字），导致同一句话
    被存进两块。此时两块内容高度重合，答案在哪块都完整。
    """
    if len(texts) < 2:
        return False
    for i in range(len(texts) - 1):
        for j in range(i + 1, len(texts)):
            m = difflib.SequenceMatcher(
                None, texts[i], texts[j], autojunk=False
            ).find_longest_match(0, len(texts[i]), 0, len(texts[j]))
            seg = texts[i][m.a:m.a + m.size]
            if m.size < min_len or not all(k in seg for k in keys):
                return False
    return True


def main():
    idx = load_json(INDEX_PATH, "索引文件 index.json")
    if idx is None:
        return
    records = idx.get("records") if isinstance(idx, dict) else idx
    if not isinstance(records, list):
        print("[错误] index.json 的结构不认识，期望 {'records': [...]}")
        return

    items = load_json(EVAL_PATH, "评估集 data/eval_set.json")
    if items is None:
        return
    if not isinstance(items, list):
        print("[错误] eval_set.json 最外层必须是方括号 [ ]，不能是花括号 { }")
        return

    print("读的题库：data\\%s" % os.path.basename(EVAL_PATH))
    print("索引：%d 块    评估集：%d 条" % (len(records), len(items)))
    print("=" * 60)

    ok_num = 0
    bad_num = 0
    type_counter = collections.Counter()

    for i, item in enumerate(items, 1):
        if not isinstance(item, dict):
            print()
            print("--- 第 %d 条  [问题] 这一条不是 {...} 格式" % i)
            bad_num += 1
            continue

        question = item.get("question", "(缺少 question 字段)")
        qtype = item.get("type", "answerable")
        keys = item.get("must_contain")
        type_counter[qtype] += 1

        print()
        print("--- 第 %d 条  [%s]  %s" % (i, qtype, question))

        if qtype == "unanswerable":
            print("    不可答题：不需要 must_contain，跳过检查")
            continue

        if not keys:
            print("    [问题] 没有 must_contain，机器没法判分")
            bad_num += 1
            continue

        if isinstance(keys, str):
            print("    [问题] must_contain 要写成 [\"词\"]，不能用单个字符串")
            bad_num += 1
            continue

        hits = [r for r in records if all(k in r.get("text", "") for k in keys)]

        if len(hits) == 1:
            print("    [OK]   唯一命中：%s" % hits[0].get("id"))
            ok_num += 1
        elif not hits:
            print("    [零]   没有任何一块同时含齐这些词：%s" % " | ".join(keys))
            for k in keys:
                print("           单独看「%s」：出现在 %d 块里" % (k, count_blocks(records, k)))
            print("           可能 1：某个词抄错了（错别字、全角半角）")
            print("           可能 2：两个词被切到了不同的块里")
            bad_num += 1
        else:
            windows = [answer_window(r.get("text", ""), keys) for r in hits]
            overlapped = is_overlap_repeat([r.get("text", "") for r in hits], keys)
            if overlapped:
                print("    [OK·重叠] 命中 %d 块，但它们是同一句话被切分 overlap 存了两遍：" % len(hits))
                for r in hits:
                    print("           %s" % r.get("id"))
                print("           答案句：「%s」" % windows[0].strip().replace("\n", " ")[:70])
                print("           -> 命中哪一块答案都对，这条可以保留")
                ok_num += 1
            else:
                print("    [多处] 命中 %d 块 —— 答错也会判你过，要换词：" % len(hits))
                for r in hits:
                    print("           %s" % r.get("id"))
                print("           这些块讲的是不同内容（不是 overlap 复制）：")
                for r, w in zip(hits, windows):
                    print("             %s : %s" % (r.get("id"), w.strip().replace("\n", " ")[:46]))
                for k in keys:
                    print("           单独看「%s」：出现在 %d 块里" % (k, count_blocks(records, k)))
                print("           改法：换一个人名 / 编号 / 生僻说法，只在一块里有的")
                bad_num += 1

    print()
    print("=" * 60)
    print("过关 %d 条，要改 %d 条" % (ok_num, bad_num))
    print("题型分布：%s" % dict(type_counter))

    total = len(items)
    if total < 50:
        print()
        print("规模建议：可答题 25 + 同义改写 15 + 不可答 10 = 50 条")
        print("当前 %d 条，还差 %d 条（样本过小，Recall 的置信区间会很宽）"
              % (total, 50 - total))


if __name__ == "__main__":
    main()
