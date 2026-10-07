"""RAG 主流程：检索 → 拼 Prompt → 生成 → 带引用返回。

【RAG 一句话】
模型自己不知道你公司的文档，所以先去你的文档里「翻」几段相关的，
连同问题一起塞给模型，让它「照着材料回答」。
检索（Retrieval）+ 生成（Generation）= RAG。

【这个文件里最重要的三个设计】
1. 拒答阈值：最高相似度低于阈值，直接回「知识库里没有」，
   不要放模型自由发挥——幻觉是 RAG 最大的敌人。
2. 强制引用：要求模型用 [1][2] 标注依据，接口把原文一起返回。
   用户能顺着编号回查，幻觉一眼可见，这也叫「可溯源」。
3. 上下文不入历史：检索到的文档片段只在本轮用，不写进会话历史，
   否则历史会被大段原文撑爆。
"""

from __future__ import annotations

from collections.abc import Iterator

from app import llm
from app.config import settings
from app.embeddings import embed_query
from app.session import SessionStore
from app.vector_store import Hit, VectorStore

SYSTEM_PROMPT = """你是一个严谨的知识库问答助手。请严格遵守以下规则：

1. 只依据【参考资料】回答，不要使用你自己的知识补充或推测。
2. 如果参考资料里没有能回答问题的内容，直接回答「知识库中没有找到相关内容」，不要编造。
3. 回答时用 [编号] 标注依据来源，例如：年假为 5 天[1]。
4. 用简洁的中文回答，先给结论，再给必要说明。"""


class RAGService:
    def __init__(
        self,
        store: VectorStore,
        sessions: SessionStore,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> None:
        self.store = store
        self.sessions = sessions
        self.top_k = top_k or settings.top_k
        self.score_threshold = (
            settings.score_threshold if score_threshold is None else score_threshold
        )

    # ---------------------------------------------------------------- 检索

    def retrieve(self, question: str, top_k: int | None = None) -> list[Hit]:
        """混合检索：向量 + 关键词，再融合排序。"""
        if len(self.store) == 0:
            return []
        query_vec = embed_query(question)
        return self.store.hybrid_search(
            query=question,
            query_vec=query_vec,
            top_k=top_k or self.top_k,
            alpha=settings.hybrid_alpha,
        )

    def is_answerable(self, hits: list[Hit]) -> bool:
        """最高分都低于阈值，说明知识库里真的没有——那就别让模型猜。"""
        return bool(hits) and max(hit.score for hit in hits) >= self.score_threshold

    @staticmethod
    def build_context(hits: list[Hit]) -> str:
        """把召回片段编号拼成参考资料。编号就是后面引用的依据。"""
        blocks = []
        for i, hit in enumerate(hits, start=1):
            blocks.append(f"[{i}] 来源：{hit.source}\n{hit.text}")
        return "\n\n".join(blocks)

    def build_messages(
        self, question: str, hits: list[Hit], history: list[dict]
    ) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append(
            {
                "role": "user",
                "content": (
                    f"【参考资料】\n{self.build_context(hits)}\n\n"
                    f"【问题】\n{question}"
                ),
            }
        )
        return messages

    @staticmethod
    def citations(hits: list[Hit]) -> list[dict]:
        return [
            {
                "index": i,
                "source": hit.source,
                "chunk_index": hit.chunk_index,
                "score": round(hit.score, 4),
                "preview": hit.text[:120],
            }
            for i, hit in enumerate(hits, start=1)
        ]

    # ------------------------------------------------------------ 同步回答

    def answer(self, question: str, session_id: str = "default") -> dict:
        hits = self.retrieve(question)

        if not self.is_answerable(hits):
            answer = "知识库中没有找到相关内容，建议换个说法或补充相关文档。"
            self.sessions.append(session_id, "user", question)
            self.sessions.append(session_id, "assistant", answer)
            return {"answer": answer, "citations": [], "grounded": False}

        messages = self.build_messages(
            question, hits, self.sessions.to_llm_messages(session_id)
        )
        answer = llm.chat(messages)["content"]

        # 只把「问题 + 回答」写进历史，参考资料不入历史
        self.sessions.append(session_id, "user", question)
        self.sessions.append(session_id, "assistant", answer)

        return {
            "answer": answer,
            "citations": self.citations(hits),
            "grounded": True,
        }

    # ------------------------------------------------------------ 流式回答

    def answer_stream(
        self, question: str, session_id: str = "default"
    ) -> Iterator[dict]:
        """产出事件字典，由 API 层转成 SSE。事件类型：citations / delta / done。"""
        hits = self.retrieve(question)

        if not self.is_answerable(hits):
            text = "知识库中没有找到相关内容，建议换个说法或补充相关文档。"
            yield {"type": "citations", "items": []}
            yield {"type": "delta", "text": text}
            yield {"type": "done"}
            self.sessions.append(session_id, "user", question)
            self.sessions.append(session_id, "assistant", text)
            return

        # 先把引用发出去：前端可以立刻渲染「正在参考这些资料」，等待感更短
        yield {"type": "citations", "items": self.citations(hits)}

        messages = self.build_messages(
            question, hits, self.sessions.to_llm_messages(session_id)
        )

        collected: list[str] = []
        for piece in llm.chat_stream(messages):
            collected.append(piece)
            yield {"type": "delta", "text": piece}

        yield {"type": "done"}

        self.sessions.append(session_id, "user", question)
        self.sessions.append(session_id, "assistant", "".join(collected))
