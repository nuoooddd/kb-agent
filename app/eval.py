"""检索效果评估：算出 Top-K 召回率和 MRR。

【为什么需要它】
「检索效果变好了」是形容词，不可验证。有了固定评估集之后，
每次改动都可以量化对比，也能避免把参数调到只对某几条题过拟合。
评估集建立后应当作为回归基准长期保留。

用法：
    python -m app.eval                  # 完整评估（需要 .env 里的 Key）
    python -m app.eval --keyword-only   # 只评估关键词检索，不需要 Key
    python -m app.eval --set data/eval_set.json --pool 5

评估集格式（JSON 数组）：
    [
      {"question": "年假有多少天？", "must_contain": ["年假", "5 天"], "type": "answerable"},
      {"question": "公司的股票代码是什么？", "type": "unanswerable"}
    ]

判定规则（刻意保持简单，便于按自己的语料替换）：
    某条命中的片段文本里**同时包含** must_contain 列出的全部字符串，
    就算这条题在当前名次上被召回。名次越靠前，MRR 越高。

⚠️ 关于 unanswerable：它只能判断「该拒答的是否拒答了」，
   也就是答案的覆盖度，衡量不了「有没有编」。要量化幻觉需要人工抽样，
   或引入 LLM 打分，当前未实现。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings  # noqa: E402
from app.vector_store import VectorStore  # noqa: E402

DEFAULT_SET = Path(settings.data_dir) / "eval_set.json"
KS = (1, 3, 5)


# --------------------------------------------------------------------- 读取

def load_cases(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"评估集不存在：{path}\n"
            "可先用自带的 data/eval_set.json 跑一遍，再替换为自己的评估集。"
        )
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise SystemExit("评估集必须是非空的 JSON 数组")
    return cases


# --------------------------------------------------------------------- 检索

def retrieve(
    store: VectorStore, question: str, keyword_only: bool, pool: int
) -> list:
    """取回候选片段。pool 要大于等于你想报告的最大 K。"""
    if keyword_only:
        return store.keyword_search(question, top_k=pool)
    # 只在完整模式里 import，保证 --keyword-only 时不依赖 openai
    from app.embeddings import embed_query

    return store.hybrid_search(
        question,
        embed_query(question),
        top_k=pool,
        alpha=settings.hybrid_alpha,
    )


def match_rank(hits: list, must_contain: list[str]) -> int | None:
    """第一个同时包含全部关键词的片段排第几（从 1 开始）；没命中返回 None。"""
    for rank, hit in enumerate(hits, start=1):
        if all(key in hit.text for key in must_contain):
            return rank
    return None


# --------------------------------------------------------------------- 主流程

def main() -> int:
    parser = argparse.ArgumentParser(description="评估检索效果")
    parser.add_argument("--set", default=str(DEFAULT_SET), help="评估集 JSON 路径")
    parser.add_argument("--index", default=settings.index_path, help="索引路径")
    parser.add_argument(
        "--keyword-only",
        action="store_true",
        help="只测关键词检索，不需要 API Key（先把脚本跑通再用它）",
    )
    parser.add_argument("--pool", type=int, default=5, help="每种检索各取多少条候选")
    args = parser.parse_args()

    store = VectorStore(path=args.index)
    if not store.load():
        print(f"索引不存在：{args.index}")
        print("先执行：python -m app.ingest")
        return 1
    if len(store) == 0:
        print("索引是空的，先执行：python -m app.ingest")
        return 1

    cases = load_cases(Path(args.set))
    answerable = [c for c in cases if c.get("type", "answerable") == "answerable"]
    unanswerable = [c for c in cases if c.get("type") == "unanswerable"]

    mode = "仅关键词（BM25）" if args.keyword_only else "混合检索（向量+BM25）"
    print(f"索引 {len(store)} 条 ｜ 评估集 {len(cases)} 条 ｜ 模式：{mode}")
    print("-" * 68)

    hits_at = {k: 0 for k in KS}
    reciprocal: list[float] = []
    details: list[dict] = []

    for case in answerable:
        question = case["question"]
        must = case.get("must_contain") or []
        hits = retrieve(store, question, args.keyword_only, args.pool)
        rank = match_rank(hits, must) if must else None

        for k in KS:
            if rank is not None and rank <= k:
                hits_at[k] += 1
        reciprocal.append(1.0 / rank if rank else 0.0)

        flag = "命中" if rank else "未命中"
        print(f"[{flag}] rank={rank if rank else '-'}  {question}")
        details.append({"question": question, "rank": rank, "must_contain": must})

    total = len(answerable)
    print("-" * 68)
    if not total:
        print("评估集里没有 answerable 的题，无法计算召回率")
        return 1

    report = {
        "mode": "keyword_only" if args.keyword_only else "hybrid",
        "index_size": len(store),
        "case_count": len(cases),
        "answerable_count": total,
        # 语料小于 30 条时指标不具统计意义
        "small_corpus": len(store) < 30,
        "top_k_recall": {f"top{k}": round(hits_at[k] / total, 4) for k in KS},
        "mrr": round(sum(reciprocal) / total, 4),
        "details": details,
    }

    for k in KS:
        print(f"Recall@{k} = {hits_at[k]}/{total} = {hits_at[k] / total:.1%}")
    print(f"MRR       = {report['mrr']:.4f}")

    # 拒答准确率：只有完整模式才有意义（关键词分数没归一化，不能和阈值比）
    if unanswerable:
        if args.keyword_only:
            print("拒答准确率：仅关键词模式下无法计算（需要归一化后的融合分）")
            report["refusal_accuracy"] = None
        else:
            ok = 0
            for case in unanswerable:
                hits = retrieve(store, case["question"], False, args.pool)
                top = max((h.score for h in hits), default=0.0)
                if top < settings.score_threshold:
                    ok += 1
                else:
                    print(f"  ⚠️ 本应拒答但分数达到 {top:.3f}：{case['question']}")
            report["refusal_accuracy"] = round(ok / len(unanswerable), 4)
            print(
                f"拒答准确率 = {ok}/{len(unanswerable)} = "
                f"{ok / len(unanswerable):.1%}（阈值 {settings.score_threshold}）"
            )

    out = Path(args.index).parent / "eval_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入 {out}")

    # 小语料上的高分不具备统计意义，必须明确提示
    if len(store) < 30:
        print(f"⚠️ 当前索引只有 {len(store)} 条，语料规模太小，上面的指标没有参考价值。")
        print("   建议先把 data/ 扩充到几十篇文档，评估集扩到 50 条")
        print("   （含同义改写题与不可答题），再重新评估。")
    else:
        print("以上指标可作为检索效果的回归基准。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
