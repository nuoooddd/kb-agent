"""核心逻辑的单元测试（chunker + vector_store，只依赖标准库）。

跑法二选一：
    pytest tests -v
    python tests/test_core.py          # 不装 pytest 也能跑

这两个模块不依赖网络，可以在没有任何密钥的情况下完整验证。
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.chunker import split_document, split_text  # noqa: E402
from app.vector_store import VectorStore, cosine, tokenize  # noqa: E402


# --------------------------------------------------------------------- 切分

def test_split_short_text_returns_single_chunk():
    assert split_text("很短的一句话。") == ["很短的一句话。"]


def test_split_empty_text_returns_nothing():
    assert split_text("") == []
    assert split_text("   \n  ") == []


def test_split_respects_chunk_size():
    text = "。".join(f"这是第{i}个句子，用来测试切分长度" for i in range(200))
    chunks = split_text(text, chunk_size=200, overlap=0)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks), [len(c) for c in chunks]


def test_split_overlap_shares_text_between_neighbours():
    text = "。".join(f"第{i}句内容" for i in range(60))
    plain = split_text(text, chunk_size=120, overlap=0)
    overlapped = split_text(text, chunk_size=120, overlap=30)
    assert len(plain) > 1 and len(overlapped) > 1
    # 加 overlap 后，每个 chunk 都比原来长
    assert len(overlapped[0]) == len(plain[0])
    assert len(overlapped[1]) > len(plain[1])
    # 关键：上一段的尾巴真的出现在下一段的开头
    assert plain[0][-30:] == overlapped[1][:30]


def test_split_handles_text_without_punctuation():
    chunks = split_text("啊" * 1000, chunk_size=100, overlap=0)
    assert len(chunks) == 10
    assert all(len(c) == 100 for c in chunks)


def test_split_document_carries_source_and_index():
    chunks = split_document("。".join(f"句子{i}" for i in range(80)), source="手册.pdf")
    assert chunks
    assert all(c.source == "手册.pdf" for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert chunks[0].id == "手册.pdf#0"


# ----------------------------------------------------------------- 向量检索

def _fake_vec(*values: float) -> list[float]:
    return list(values)


def test_cosine_basics():
    assert cosine([1, 0], [1, 0]) == 1.0
    assert abs(cosine([1, 0], [0, 1])) < 1e-9
    assert cosine([], [1, 0]) == 0.0
    assert cosine([1, 0], [1, 0, 0]) == 0.0  # 维度不一致要安全返回 0


def test_tokenize_splits_latin_and_cjk():
    tokens = tokenize("Spring Boot 配置")
    assert "spring" in tokens and "boot" in tokens
    assert "配" in tokens and "配置" in tokens


def test_vector_search_ranks_by_similarity():
    store = VectorStore(path=":memory:")
    store.add(
        [
            {"id": "a#0", "text": "向量检索", "source": "a", "chunk_index": 0,
             "embedding": _fake_vec(1.0, 0.0, 0.0)},
            {"id": "b#0", "text": "数据库", "source": "b", "chunk_index": 0,
             "embedding": _fake_vec(0.0, 1.0, 0.0)},
        ]
    )
    hits = store.search(_fake_vec(0.9, 0.1, 0.0), top_k=2)
    assert hits[0].id == "a#0"
    assert hits[0].score > hits[1].score


def test_keyword_search_finds_exact_term():
    store = VectorStore(path=":memory:")
    store.add(
        [
            {"id": "a#0", "text": "本项目使用 Chroma 作为向量库", "source": "a",
             "chunk_index": 0, "embedding": _fake_vec(1.0, 0.0)},
            {"id": "b#0", "text": "今天天气不错，适合出门散步", "source": "b",
             "chunk_index": 0, "embedding": _fake_vec(0.0, 1.0)},
        ]
    )
    hits = store.keyword_search("Chroma", top_k=2)
    assert hits and hits[0].id == "a#0"


def test_hybrid_search_merges_two_routes():
    store = VectorStore(path=":memory:")
    store.add(
        [
            {"id": "a#0", "text": "向量库选型对比", "source": "a", "chunk_index": 0,
             "embedding": _fake_vec(1.0, 0.0)},
            {"id": "b#0", "text": "Qdrant 支持按部门过滤", "source": "b",
             "chunk_index": 0, "embedding": _fake_vec(0.0, 1.0)},
        ]
    )
    hits = store.hybrid_search("Qdrant", _fake_vec(1.0, 0.0), top_k=2)
    ids = {h.id for h in hits}
    assert ids == {"a#0", "b#0"}  # 两路都召回，融合后都在
    assert all(0.0 <= h.score <= 1.0 for h in hits)


def test_store_save_and_load_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "index.json")
        store = VectorStore(path=path)
        store.add([{"id": "a#0", "text": "持久化", "source": "a", "chunk_index": 0,
                    "embedding": _fake_vec(1.0, 0.0)}])
        store.save()

        again = VectorStore(path=path)
        assert again.load() is True
        assert len(again) == 1
        assert again.search(_fake_vec(1.0, 0.0), top_k=1)[0].text == "持久化"


def test_store_load_missing_file_returns_false():
    assert VectorStore(path="不存在的文件.json").load() is False


if __name__ == "__main__":
    import traceback

    failed = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception:  # noqa: BLE001
                failed += 1
                print(f"FAIL  {name}")
                traceback.print_exc()
    print(f"\n{'全部通过' if not failed else f'{failed} 个用例失败'}")
    sys.exit(1 if failed else 0)
