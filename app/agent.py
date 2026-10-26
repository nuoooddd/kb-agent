"""Agent 层：让模型自己决定「要不要查资料、查什么」。

【RAG 和 Agent 的区别】
- RAG 是固定流程：先检索，再生成。检索这一步是代码写死的，模型没得选。
- Agent 是模型主导：把「工具」描述给模型，由它决定调哪个、传什么参数、
  要不要连着调好几个。代码只负责执行工具并把结果喂回去。

【多轮 tool loop 的流程】
    用户提问
      → 模型返回 tool_calls（说要查知识库）
      → 我们执行工具，把结果作为 role=tool 的消息塞回去
      → 模型拿到结果，再决定是继续调工具，还是给出最终答案
      → 直到它不再调工具，或到达 max_rounds 上限

【必须处理的三件事，否则线上必出问题】
1. 死循环：模型可能反复调同一个工具。所以设轮数上限 + 重复调用短路。
2. 工具报错：工具挂了不能让整个请求崩，要把错误信息当结果返回给模型，
   让模型自己决定重试还是如实说「查询失败」。
3. 参数不可信：模型给的参数是自然语言猜的，必须校验（这里是 top_k 夹取范围）。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta

from app import llm
from app.config import settings
from app.rag import SYSTEM_PROMPT, RAGService

# 东八区，用于「现在几点」这个工具
CST = timezone(timedelta(hours=8))

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "在私有知识库中检索与问题相关的文档片段。"
                "当问题涉及公司制度、产品文档、项目资料等私有内容时必须调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索用的关键词或问句，尽量保留原问题的关键名词",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回条数，默认 4，最大 10",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期和时间。当问题涉及「今天」「现在」这类相对时间时调用。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

# 给模型看的工具说明，放在 system prompt 后面
AGENT_PROMPT = SYSTEM_PROMPT + """

你可以调用工具来获取信息：
- 需要私有资料时，调用 search_knowledge_base；
- 需要当前时间时，调用 get_current_time。
调用工具后，请基于返回结果回答，并保留 [编号] 引用格式。
"""


class KnowledgeAgent:
    def __init__(self, rag: RAGService, max_rounds: int | None = None) -> None:
        self.rag = rag
        self.max_rounds = max_rounds or settings.max_agent_rounds

    # ------------------------------------------------------------- 工具执行

    def _exec_search(self, args: dict) -> str:
        query = str(args.get("query", "")).strip()
        if not query:
            return json.dumps({"error": "query 不能为空"}, ensure_ascii=False)

        # 模型给的数字不可信，夹到合理范围
        try:
            top_k = int(args.get("top_k", 4))
        except (TypeError, ValueError):
            top_k = 4
        top_k = max(1, min(top_k, 10))

        hits = self.rag.retrieve(query, top_k=top_k)
        if not hits:
            return json.dumps({"results": [], "note": "没有检索到内容"}, ensure_ascii=False)

        return json.dumps(
            {
                "results": [
                    {
                        "source": hit.source,
                        "score": round(hit.score, 4),
                        "text": hit.text,
                    }
                    for hit in hits
                ]
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _exec_time(_args: dict) -> str:
        return json.dumps(
            {"now": datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S"), "timezone": "UTC+8"},
            ensure_ascii=False,
        )

    def _run_tool(self, name: str, args: dict) -> str:
        # 工具内部异常不向上抛：转成文本结果交回模型，让它自己决定怎么办
        try:
            if name == "search_knowledge_base":
                return self._exec_search(args)
            if name == "get_current_time":
                return self._exec_time(args)
            return json.dumps({"error": f"未知工具：{name}"}, ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"error": f"工具执行失败：{exc}"}, ensure_ascii=False)

    # ----------------------------------------------------------------- 主循环

    def run(self, question: str, session_id: str = "default") -> dict:
        messages: list[dict] = [
            {"role": "system", "content": AGENT_PROMPT},
            *self.rag.sessions.to_llm_messages(session_id),
            {"role": "user", "content": question},
        ]

        citations: list[dict] = []
        tool_log: list[str] = []
        seen_calls: set[tuple[str, str]] = set()  # 防重复调用
        reply = {"content": "", "tool_calls": []}

        for _round in range(self.max_rounds):
            reply = llm.chat(messages, tools=TOOLS)
            if not reply["tool_calls"]:
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": reply["content"],
                    "tool_calls": reply["tool_calls"],
                }
            )

            for call in reply["tool_calls"]:
                name = call["function"]["name"]
                raw_args = call["function"].get("arguments") or "{}"
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {}

                signature = (name, json.dumps(args, sort_keys=True, ensure_ascii=False))
                if signature in seen_calls:
                    # 同一轮里重复调同样的参数，直接短路，避免烧 token
                    result = json.dumps(
                        {"note": "该工具调用已执行过，请直接基于已有结果回答"},
                        ensure_ascii=False,
                    )
                else:
                    seen_calls.add(signature)
                    result = self._run_tool(name, args)
                    tool_log.append(name)

                messages.append(
                    {"role": "tool", "tool_call_id": call["id"], "content": result}
                )

        else:
            # 用完所有轮次还在调工具：强制关掉工具再问一次，保证一定给出回答
            reply = llm.chat(messages)

        answer = reply["content"] or "抱歉，我没能完成这次查询。"

        self.rag.sessions.append(session_id, "user", question)
        self.rag.sessions.append(session_id, "assistant", answer)

        return {
            "answer": answer,
            "tool_calls": tool_log,
            "rounds": len(tool_log),
            "citations": citations,
        }
