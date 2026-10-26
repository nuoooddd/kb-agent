"""离线入库脚本：把 data/ 目录里的文档解析 → 切分 → 向量化 → 存成索引。

用法：
    python -m app.ingest                  # 处理 data/ 下所有文档
    python -m app.ingest ./docs           # 指定目录
    python -m app.ingest --dry-run        # 只切分不向量化，不需要 API Key

--dry-run 用于在申请密钥之前先验证切分效果；
调 chunk_size 时反复跑这个，比每次都调用向量化接口划算得多。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.chunker import Chunk, split_document  # noqa: E402
from app.config import settings  # noqa: E402
from app.vector_store import VectorStore  # noqa: E402

TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
PDF_SUFFIXES = {".pdf"}
DOCX_SUFFIXES = {".docx"}


# ---------------------------------------------------------------- 文档解析

def load_text(path: Path) -> str:
    """把各种格式的文档读成纯文本。缺依赖时给出清楚的提示，而不是崩一堆栈。"""
    suffix = path.suffix.lower()

    if suffix in TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix in PDF_SUFFIXES:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError(
                f"解析 {path.name} 需要 pypdf，请先执行：pip install pypdf"
            ) from exc
        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)

    if suffix in DOCX_SUFFIXES:
        try:
            import docx
        except ImportError as exc:
            raise RuntimeError(
                f"解析 {path.name} 需要 python-docx，请先执行：pip install python-docx"
            ) from exc
        document = docx.Document(str(path))
        return "\n".join(p.text for p in document.paragraphs)

    raise RuntimeError(f"暂不支持的文件类型：{suffix}")


def collect_files(root: Path) -> list[Path]:
    allowed = TEXT_SUFFIXES | PDF_SUFFIXES | DOCX_SUFFIXES
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in allowed
    )


def build_chunks(root: Path, chunk_size: int, overlap: int) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for path in collect_files(root):
        try:
            text = load_text(path)
        except RuntimeError as exc:
            print(f"  跳过 {path.name}：{exc}")
            continue

        chunks = split_document(
            text,
            source=path.name,
            chunk_size=chunk_size,
            overlap=overlap,
            metadata={"path": str(path)},
        )
        print(f"  {path.name}: {len(text)} 字 → {len(chunks)} 个 chunk")
        all_chunks.extend(chunks)
    return all_chunks


# ------------------------------------------------------------------- 主流程

def main() -> int:
    parser = argparse.ArgumentParser(description="把文档灌进知识库索引")
    parser.add_argument("data_dir", nargs="?", default=settings.data_dir, help="文档目录")
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--overlap", type=int, default=settings.chunk_overlap)
    parser.add_argument("--index", default=settings.index_path, help="索引输出路径")
    parser.add_argument("--dry-run", action="store_true", help="只切分，不调向量化接口")
    parser.add_argument(
        "--no-embed",
        action="store_true",
        help="先不向量化，生成一个只能跑关键词检索的索引（用于没配 Key 时先跑评估）",
    )
    args = parser.parse_args()

    root = Path(args.data_dir)
    if not root.exists():
        print(f"目录不存在：{root}")
        return 1

    files = collect_files(root)
    if not files:
        print(f"{root} 里没有可解析的文档（支持 .txt/.md/.pdf/.docx）")
        return 1

    print(f"扫描到 {len(files)} 个文件，chunk_size={args.chunk_size} overlap={args.overlap}")
    chunks = build_chunks(root, args.chunk_size, args.overlap)
    if not chunks:
        print("没有切出任何 chunk，检查一下文档内容")
        return 1

    lengths = [len(c.text) for c in chunks]
    print(
        f"共 {len(chunks)} 个 chunk，平均 {sum(lengths) // len(lengths)} 字，"
        f"最长 {max(lengths)} 字，最短 {min(lengths)} 字"
    )

    if args.dry_run:
        print("\n[--dry-run] 预览前 2 个 chunk：")
        for chunk in chunks[:2]:
            print(f"\n--- {chunk.id} ---\n{chunk.text[:300]}")
        print("\n没有写入索引。去掉 --dry-run 并配好 .env 后会真正向量化入库。")
        return 0

    if args.no_embed:
        # 没配 Key 时也能有一份能用的索引：只跑关键词那一路。
        # 之后配好 Key 重新执行一次 python -m app.ingest 就会被完整索引覆盖。
        print("\n[--no-embed] 跳过向量化，只写文本（此索引仅支持关键词检索）")
        vectors = [[] for _ in chunks]
    else:
        # 向量化：放在这里 import，dry-run 时不需要装 openai
        from app.embeddings import embed_texts

        print(f"\n开始向量化 {len(chunks)} 个 chunk（模型 {settings.embedding_model}）……")
        vectors = embed_texts([c.text for c in chunks])

    records = [
        {
            "id": chunk.id,
            "text": chunk.text,
            "source": chunk.source,
            "chunk_index": chunk.chunk_index,
            "metadata": chunk.metadata,
            "embedding": vector,
        }
        for chunk, vector in zip(chunks, vectors)
    ]

    store = VectorStore(path=args.index)
    store.add(records)
    store.save()
    print(f"\n索引已写入 {args.index}（{len(store)} 条）")
    if args.no_embed:
        print("提示：现在可以跑 python -m app.eval --keyword-only 看关键词召回率；")
        print("      配好 .env 后再执行一次 python -m app.ingest 换成完整索引。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
