"""向量库 + 混合检索（纯标准库实现）。

【为什么先自己写一个，而不是一上来装 Chroma / Qdrant】
1. 装 Chroma 会带一堆依赖，容易在第一步就卡住；
2. 余弦相似度、BM25、混合权重归一化这几处需要反复调参，
   自研实现能直接改内部逻辑，不用绕开库的封装；
3. 对外接口与 Qdrant / Chroma 对齐，数据量上千、需要元数据过滤时，
   可以替换实现而不改动调用方。

【存的是什么】
每条记录 = {id, text, source, chunk_index, embedding}
    id          引用溯源用，形如 "员工手册.pdf#3"
    embedding   向量，一个 float 列表
整份索引用一个 JSON 文件持久化（小规模够用，上千条以上就该换真正的向量库了）。

【三种检索方式，建议按顺序理解】
1. search()          纯向量：语义相似，但专有名词 / 编号 / 缩写会漏
2. keyword_search()  BM25：按词命中，专有名词很准，但不会「理解意思」
3. hybrid_search()   两者加权融合，先各自归一化再相加，工程上最常用
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass, field

_LATIN = re.compile(r"[a-zA-Z0-9_]+")
_CJK = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """中英混排的极简分词器（不装 jieba 也能用）。

    中文没有空格，标准做法是装分词库。这里用「字符二元组」近似：
    「知识库检索」→ 知,识,库,检,索,知识,识库,库检,检索
    好处是零依赖、专有名词召回还不错；坏处是词表膨胀，长文档要换真分词器。
    """
    text = (text or "").lower()
    tokens = _LATIN.findall(text)
    cjk = _CJK.findall(text)
    tokens.extend(cjk)
    tokens.extend(a + b for a, b in zip(cjk, cjk[1:]))
    return tokens


def cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度：只看向量方向，不受长度影响，是文本检索的默认选择。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@dataclass
class Hit:
    """一条检索结果。三个分数分开留存，便于定位是向量路还是关键词路召回。"""

    id: str
    text: str
    source: str
    chunk_index: int
    score: float
    vec_score: float = 0.0
    kw_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VectorStore:
    path: str = "index.json"
    records: list[dict] = field(default_factory=list)

    # ---- 以下都是 BM25 的懒加载缓存，不参与持久化 ----
    _tf: list[Counter] = field(default_factory=list, repr=False)
    _df: dict[str, int] = field(default_factory=dict, repr=False)
    _lens: list[int] = field(default_factory=list, repr=False)
    _avg_len: float = 1.0
    _stats_ready: bool = field(default=False, repr=False)

    # ------------------------------------------------------------------ 写入

    def add(self, records: list[dict]) -> None:
        """追加记录。每条记录必须含 embedding，否则检索时会算不出分。"""
        self.records.extend(records)
        self._stats_ready = False

    def __len__(self) -> int:
        return len(self.records)

    def save(self) -> None:
        parent = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(parent, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"records": self.records}, f, ensure_ascii=False)
        self._stats_ready = False

    def load(self) -> bool:
        """读回索引；不存在就返回 False，让调用方决定怎么提示。"""
        if not os.path.exists(self.path):
            return False
        with open(self.path, encoding="utf-8") as f:
            self.records = json.load(f).get("records", [])
        self._stats_ready = False
        return True

    # ------------------------------------------------------------- 分数计算

    def _ensure_stats(self) -> None:
        """一次性算好 BM25 需要的统计量：词频、文档频率、平均长度。"""
        if self._stats_ready:
            return
        self._tf = []
        self._df = {}
        self._lens = []
        for rec in self.records:
            counter = Counter(tokenize(rec.get("text", "")))
            self._tf.append(counter)
            self._lens.append(sum(counter.values()) or 1)
            for token in counter:
                self._df[token] = self._df.get(token, 0) + 1
        self._avg_len = (sum(self._lens) / len(self._lens)) if self._lens else 1.0
        self._stats_ready = True

    def _hit(self, index: int, score: float) -> Hit:
        rec = self.records[index]
        return Hit(
            id=rec.get("id", str(index)),
            text=rec.get("text", ""),
            source=rec.get("source", ""),
            chunk_index=int(rec.get("chunk_index", index)),
            score=score,
        )

    # ----------------------------------------------------------------- 检索

    def search(self, query_vec: list[float], top_k: int = 5) -> list[Hit]:
        """纯向量检索：语义匹配。"""
        scored: list[tuple[float, int]] = []
        for i, rec in enumerate(self.records):
            score = cosine(query_vec, rec.get("embedding") or [])
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda x: x[0], reverse=True)
        hits = [self._hit(i, s) for s, i in scored[:top_k]]
        for hit in hits:
            hit.vec_score = hit.score
        return hits

    def keyword_search(
        self, query: str, top_k: int = 5, k1: float = 1.5, b: float = 0.75
    ) -> list[Hit]:
        """BM25 关键词检索：专有名词、编号、缩写的克星。

        idf 的含义：一个词在越少的文档里出现，就越有区分度，权重越高。
        k1 控制词频饱和（同一个词出现 10 次不等于重要 10 倍），
        b  控制长度归一化（长文档天然更容易命中，需要惩罚）。
        """
        self._ensure_stats()
        total = len(self.records)
        if total == 0:
            return []

        scores = [0.0] * total
        for token in set(tokenize(query)):
            df = self._df.get(token, 0)
            if df == 0:
                continue
            idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
            for i, counter in enumerate(self._tf):
                tf = counter.get(token, 0)
                if tf == 0:
                    continue
                dl = self._lens[i]
                denom = tf + k1 * (1 - b + b * dl / self._avg_len)
                scores[i] += idf * (tf * (k1 + 1)) / denom

        ranked = sorted(range(total), key=lambda i: scores[i], reverse=True)
        hits: list[Hit] = []
        for i in ranked[:top_k]:
            if scores[i] <= 0:
                continue
            hit = self._hit(i, scores[i])
            hit.kw_score = hit.score
            hits.append(hit)
        return hits

    @staticmethod
    def _normalize(hits: list[Hit]) -> dict[str, float]:
        """把一路检索的原始分压到 0~1，否则向量分和 BM25 分的量纲没法相加。"""
        if not hits:
            return {}
        values = [h.score for h in hits]
        low, high = min(values), max(values)
        span = (high - low) or 1.0
        return {h.id: (h.score - low) / span for h in hits}

    def hybrid_search(
        self,
        query: str,
        query_vec: list[float],
        top_k: int = 5,
        alpha: float = 0.6,
        pool: int = 10,
    ) -> list[Hit]:
        """混合检索：alpha 是向量得分权重，(1-alpha) 给关键词得分。"""
        vec_hits = self.search(query_vec, top_k=pool)
        kw_hits = self.keyword_search(query, top_k=pool)

        vec_norm = self._normalize(vec_hits)
        kw_norm = self._normalize(kw_hits)

        merged: dict[str, Hit] = {}
        for hit in vec_hits:
            merged[hit.id] = hit
        for hit in kw_hits:
            merged.setdefault(hit.id, hit)

        results: list[Hit] = []
        for hit_id, hit in merged.items():
            hit.vec_score = vec_norm.get(hit_id, 0.0)
            hit.kw_score = kw_norm.get(hit_id, 0.0)
            hit.score = alpha * hit.vec_score + (1 - alpha) * hit.kw_score
            results.append(hit)

        results.sort(key=lambda h: h.score, reverse=True)
        return results[:top_k]
