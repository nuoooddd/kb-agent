"""MCP Server：把 kb-agent 的检索能力暴露成标准工具，供任意 MCP 客户端调用。

【它解决什么问题】
kb-agent 原来只能自己用（起 FastAPI 服务，用 curl 或网页问）。
MCP（Model Context Protocol）是让「别的 AI 客户端」能挂载并调用你的代码的开放协议。
接上之后，Claude Desktop、Cursor、Cherry Studio 这类客户端只要配一行命令，
就能把本服务当成内置工具，直接检索这个知识库。

【为什么手写协议，不装官方 SDK】
本项目检索层是纯标准库实现（自研 chunker / vector_store / BM25），
引入 mcp 包会把「零第三方依赖」这个性质破坏掉，也没法在面试里讲清协议细节。
协议本身很小：JSON-RPC 2.0，一行一条消息，走 stdin / stdout。
服务端必须处理的消息只有四类：

    initialize          握手，回协议版本与能力声明
    tools/list          报工具清单（名字、说明、参数 JSON Schema）
    tools/call          执行一次工具调用
    notifications/*     通知，没有 id，不需要回

【stdout 是独占的】
MCP 规定 stdout 只能出现 JSON-RPC 消息。任何 print 都会污染协议流，
客户端解析失败会直接断开。所以本文件所有日志一律写 stderr。
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.config import settings  # noqa: E402
from app.rag import RAGService  # noqa: E402
from app.session import SessionStore  # noqa: E402
from app.vector_store import VectorStore  # noqa: E402

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "kb-agent", "version": "1.0.0"}

# ---------------------------------------------------------------- 工具定义

TOOLS = [
    {
        "name": "search_knowledge",
        "description": (
            "在 Prompt Engineering 知识库中做混合检索（向量 + BM25），"
            "只返回原文片段与出处，不生成答案。"
            "适合需要自己组织回答、或要核对原文出处的场景。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索问题或关键词"},
                "top_k": {
                    "type": "integer",
                    "description": "返回片段数，默认取项目配置（4），上限 10",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "ask_knowledge_base",
        "description": (
            "完整 RAG 问答：先检索，再让大模型仅依据检索片段作答，"
            "答案带 [1][2] 引用编号。知识库没有相关内容时明确拒答，不编造。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "用户问题"}},
            "required": ["query"],
        },
    },
]


class KnowledgeBase:
    """索引 + 会话 + RAG 服务的装配。懒加载：第一次用到工具时才读索引。"""

    def __init__(self) -> None:
        self._rag: RAGService | None = None

    def rag(self) -> RAGService:
        if self._rag is None:
            store = VectorStore(path=settings.index_path)
            if not store.load():
                raise RuntimeError(
                    f"索引不存在：{settings.index_path}。"
                    "先运行 app.ingest 把文档灌进索引。"
                )
            sessions = SessionStore(
                redis_url=settings.redis_url,
                max_turns=settings.max_turns,
                ttl=settings.session_ttl,
            )
            self._rag = RAGService(store=store, sessions=sessions)
        return self._rag


KB = KnowledgeBase()


# ---------------------------------------------------------------- 工具实现


def search_knowledge(args: dict) -> str:
    query = (args.get("query") or "").strip()
    if not query:
        raise ValueError("query 不能为空")

    top_k = args.get("top_k") or settings.top_k
    top_k = max(1, min(int(top_k), 10))

    rag = KB.rag()
    hits = rag.retrieve(query, top_k=top_k)
    if not hits:
        return "知识库为空，没有可检索的内容。"

    answerable = rag.is_answerable(hits)
    best = max(hit.vec_raw for hit in hits)
    lines = [
        f"命中 {len(hits)} 条｜最高原始余弦 {best:.4f}｜拒答阈值 {rag.score_threshold:.2f}"
        f"｜判定：{'可答' if answerable else '低于阈值，知识库可能没有相关内容'}",
        "",
    ]
    for i, hit in enumerate(hits, start=1):
        lines.append(
            f"[{i}] {hit.source} #{hit.chunk_index}"
            f"　融合分 {hit.score:.4f} / 余弦 {hit.vec_raw:.4f}"
        )
        lines.append(hit.text)
        lines.append("")
    return "\n".join(lines).rstrip()


def ask_knowledge_base(args: dict) -> str:
    query = (args.get("query") or "").strip()
    if not query:
        raise ValueError("query 不能为空")

    result = KB.rag().answer(query, session_id="mcp")
    lines = [result.get("answer", "").strip(), ""]
    for cite in result.get("citations") or []:
        lines.append(
            f"[{cite['index']}] {cite['source']} #{cite['chunk_index']}"
            f"　融合分 {cite['score']:.4f}"
        )
    return "\n".join(lines).rstrip()


HANDLERS = {
    "search_knowledge": search_knowledge,
    "ask_knowledge_base": ask_knowledge_base,
}


# ---------------------------------------------------------------- 协议分发


def handle(req: dict) -> dict | None:
    """处理一条请求，返回响应；通知类返回 None（不回）。"""
    method = req.get("method")
    req_id = req.get("id")

    # 通知没有 id，按协议不能有响应
    if req_id is None and method and method.startswith("notifications/"):
        return None

    if method == "initialize":
        result = {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": SERVER_INFO,
        }
        return ok(req_id, result)

    if method == "ping":
        return ok(req_id, {})

    if method == "tools/list":
        return ok(req_id, {"tools": TOOLS})

    if method == "tools/call":
        params = req.get("params") or {}
        name = params.get("name")
        args = params.get("arguments") or {}
        handler = HANDLERS.get(name)
        if handler is None:
            return err(req_id, -32602, f"未知工具：{name}")
        try:
            text = handler(args)
        except Exception as exc:  # 工具内部异常要回给客户端，不能让它干等
            return ok(
                req_id,
                {"content": [{"type": "text", "text": f"调用失败：{exc}"}], "isError": True},
            )
        return ok(req_id, {"content": [{"type": "text", "text": text}], "isError": False})

    return err(req_id, -32601, f"不支持的方法：{method}")


def ok(req_id, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def err(req_id, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def send(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- 入口


def serve() -> int:
    log(f"MCP server 已启动（stdio）｜索引 {settings.index_path}")
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            send(err(None, -32700, f"JSON 解析失败：{exc}"))
            continue
        resp = handle(req)
        if resp is not None:
            send(resp)
    return 0


def selftest() -> int:
    """不接客户端，自己走一遍完整协议流，用于验证与演示。"""
    print("=" * 68)
    print("MCP 自测：模拟一个客户端，按协议顺序发 4 条消息")
    print("=" * 68)

    steps = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {}, "clientInfo": {"name": "selftest", "version": "0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "search_knowledge",
                    "arguments": {"query": "自我一致性为什么能提升准确率", "top_k": 2}}},
    ]

    for step in steps:
        label = step.get("method") or "?"
        print(f"\n>>> [{step.get('id')}] {label}")
        print(json.dumps(step, ensure_ascii=False)[:160])

        resp = handle(step)
        if resp is None:
            print("<<< （通知，按协议不返回）")
            continue
        text = json.dumps(resp, ensure_ascii=False)
        print("<<< " + (text if len(text) <= 700 else text[:700] + " …"))

    print("\n" + "=" * 68)
    print("自测结束。四项全部有回应即协议通。")
    return 0


if __name__ == "__main__":
    # Windows 控制台默认不是 UTF-8，不显式指定会让中文变成乱码
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    if "--selftest" in sys.argv:
        sys.exit(selftest())
    sys.exit(serve())
