"""文本切分（chunking）：把长文档切成适合检索的小段。

【为什么不用现成的切分库】
LangChain 的 `RecursiveCharacterTextSplitter` 一行就能用，但切分策略直接决定召回质量，
分隔符列表和合并阈值需要按语料反复调；自己实现一遍才能控制这些细节，
也避免为一个函数引入整条依赖链。

【递归切分的思路】
1. 按「优先级从高到低」的分隔符列表逐个尝试：段落 → 换行 → 句号 → 分号 → 逗号 → 空格；
2. 先用最粗的分隔符切，切出来的片段如果还比 chunk_size 长，就换下一个更细的分隔符继续切；
3. 最后把过短的片段合并回 chunk_size 附近，并给相邻 chunk 加 overlap。

【为什么要 overlap】
一句话被切断时，语义会残缺。让相邻 chunk 共享 80 个字符的重叠区，
检索时就不会因为「关键词正好落在切口上」而漏掉内容。
代价是存储和 token 变多，所以 overlap 一般是 chunk_size 的 10%~20%。

【为什么 chunk_size 取 500】
- 太小（如 100）：语义不完整，检索到的片段没头没尾，模型用不上；
- 太大（如 2000）：一段里混了多个主题，向量被「平均」掉，相似度区分度下降，
  而且一召回就塞满上下文。中文 300~800 是比较稳的区间，500 是常见起点。

纯标准库实现，不需要装任何依赖，可以直接跑测试验证。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# 优先级从粗到细；最后的 "" 是兜底（整段没有任何标点时按长度硬切）
DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]


@dataclass
class Chunk:
    """一个切分单元，也就是最终被向量化的最小文本块。"""

    text: str
    source: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        """全局唯一 id：文件名 + 序号。用来做去重和引用溯源。"""
        return f"{self.source}#{self.chunk_index}"


def _recursive_split(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """按分隔符优先级递归切分，保证返回的片段尽量不超过 chunk_size。"""
    if len(text) <= chunk_size:
        return [text] if text else []

    sep = separators[0] if separators else ""
    # 分隔符用完时用 "" 兜底，走下面的硬切分支
    rest = separators[1:] or [""]

    if sep == "":
        # 极端情况：一整段没有标点。只能按固定长度硬切，宁可切断也不能超长。
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    parts = text.split(sep)

    # split 会把分隔符吃掉，这里补回去，避免句子之间粘连
    pieces: list[str] = []
    for i, part in enumerate(parts):
        pieces.append(part + sep if i < len(parts) - 1 else part)

    out: list[str] = []
    for piece in pieces:
        if not piece:
            continue
        if len(piece) <= chunk_size:
            out.append(piece)
        else:
            out.extend(_recursive_split(piece, chunk_size, rest))
    return out


def _merge_pieces(pieces: list[str], chunk_size: int) -> list[str]:
    """把切碎的小片段往回合并，尽量填满 chunk_size，减少碎片。"""
    merged: list[str] = []
    buf = ""
    for piece in pieces:
        if not piece:
            continue
        if len(buf) + len(piece) <= chunk_size:
            buf += piece
        else:
            if buf:
                merged.append(buf)
            buf = piece
    if buf:
        merged.append(buf)
    return merged


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    """给每个 chunk 的头部拼上一段的尾部。"""
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    out = [chunks[0]]
    for i in range(1, len(chunks)):
        tail = chunks[i - 1][-overlap:]
        out.append(tail + chunks[i])
    return out


def split_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 80,
    separators: list[str] | None = None,
) -> list[str]:
    """把一段长文本切成若干 chunk（字符串列表）。"""
    text = (text or "").strip()
    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正数")

    seps = list(separators) if separators else list(DEFAULT_SEPARATORS)
    pieces = _recursive_split(text, chunk_size, seps)
    chunks = [c.strip() for c in _merge_pieces(pieces, chunk_size) if c.strip()]
    return _apply_overlap(chunks, overlap)


def split_document(
    text: str,
    source: str,
    chunk_size: int = 500,
    overlap: int = 80,
    metadata: dict | None = None,
) -> list[Chunk]:
    """切成带元信息的 Chunk 列表，元信息用于最终答案里的引用溯源。"""
    return [
        Chunk(
            text=chunk,
            source=source,
            chunk_index=i,
            metadata=dict(metadata or {}),
        )
        for i, chunk in enumerate(split_text(text, chunk_size, overlap))
    ]
