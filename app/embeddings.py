"""向量化（embedding）：把文本变成一串浮点数。

【一句话理解】
Embedding 就是把「意思」变成一个坐标。意思相近的两句话，
在向量空间里距离就近——这就是语义检索能work的全部原理。

【为什么用 API 而不是本地模型】
本地方案（sentence-transformers + bge-small-zh）不需要联网、不花钱，
但会拉下来 PyTorch，动辄 2GB，环境准备成本高。
先用 API 把整条链路跑通，之后想换本地模型，只要替换这个文件里的函数即可——
对外只暴露 embed_texts / embed_query 两个函数，实现可替换。

【批量要注意的两件事】
1. 一次别塞太多文本，按 batch 切开发请求；
2. 返回结果的顺序不一定和输入一致，必须按 index 排回去（很多人在这里踩坑）。
"""

from __future__ import annotations

from app.config import settings

_client = None


def get_client():
    """懒加载客户端，避免没配密钥时 import 就报错。"""
    global _client
    if _client is None:
        if not (settings.embedding_api_key or settings.llm_api_key):
            raise RuntimeError(
                "没有找到 embedding 的 API Key。请在项目根目录创建 .env，"
                "参考 .env.example 填入 LLM_API_KEY 或 EMBEDDING_API_KEY。"
            )
        from openai import OpenAI  # 延迟 import，方便没装依赖时也能看到上面这句提示

        _client = OpenAI(**settings.embedding_client_kwargs())
    return _client


def embed_texts(texts: list[str], batch_size: int | None = None) -> list[list[float]]:
    """把一批文本转成向量列表，顺序与输入一致。"""
    if not texts:
        return []

    client = get_client()
    size = batch_size or settings.embedding_batch
    vectors: list[list[float]] = []

    for start in range(0, len(texts), size):
        batch = texts[start : start + size]
        resp = client.embeddings.create(model=settings.embedding_model, input=batch)
        # 按 index 排序，不能假设返回顺序 == 输入顺序
        items = sorted(resp.data, key=lambda item: item.index)
        vectors.extend(item.embedding for item in items)

    if len(vectors) != len(texts):
        raise RuntimeError(f"向量数量({len(vectors)})与文本数量({len(texts)})不一致")

    return vectors


def embed_query(text: str) -> list[float]:
    """检索时对用户的问句做向量化。

    注意：检索用的「问句」和入库用的「文档片段」必须用同一个 embedding 模型，
    否则向量空间对不上，相似度完全没意义——这是新手最常见的事故。
    """
    return embed_texts([text])[0]
