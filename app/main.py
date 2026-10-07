"""HTTP 服务入口（FastAPI）。

【为什么用 FastAPI 而不是 Flask】
1. 自带数据校验（Pydantic）：请求体字段写错了，框架直接返回 422 和原因，
   不用自己写一堆 if 判断；
2. 自动生成交互式文档：启动后打开 /docs 就能直接点着调接口，演示很方便；
3. 原生支持 async 和流式响应，做 SSE 很自然。

【一个容易被忽略的性能细节】
这里的接口都用同步 `def` 而不是 `async def`。
因为 llm / embedding 的调用是阻塞式的（用的是 requests 那套），
写成 async def 会把整个事件循环卡住，一个请求就能让服务无法响应别人。
FastAPI 对同步函数会自动丢到线程池执行，所以用同步 def 反而是对的选择。
真要用 async，就得换成异步 HTTP 客户端并全程 await。
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import HTMLResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from app.agent import KnowledgeAgent  # noqa: E402
from app.config import settings  # noqa: E402
from app.rag import RAGService  # noqa: E402
from app.session import SessionStore  # noqa: E402
from app.vector_store import VectorStore  # noqa: E402

# 全局状态：启动时装配一次，请求之间复用
state: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    store = VectorStore(path=settings.index_path)
    loaded = store.load()
    sessions = SessionStore(
        redis_url=settings.redis_url,
        max_turns=settings.max_turns,
        ttl=settings.session_ttl,
    )
    rag = RAGService(store=store, sessions=sessions)

    state.update(store=store, sessions=sessions, rag=rag, agent=KnowledgeAgent(rag))

    print(f"索引：{settings.index_path}（{'已加载' if loaded else '不存在'}，{len(store)} 条）")
    print(f"会话存储：{sessions.backend}")
    if not loaded:
        print("提示：先执行 python -m app.ingest 把文档灌进索引，再问问题。")
    yield
    state.clear()


app = FastAPI(title="知识库问答 Agent", version="0.1.0", lifespan=lifespan)


# ------------------------------------------------------------------ 请求模型

class SearchRequest(BaseModel):
    query: str = Field(min_length=1, description="检索问句")
    top_k: int = Field(default=4, ge=1, le=20)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, description="用户问题")
    session_id: str = Field(default="default", description="会话 id，用于多轮记忆")


# ---------------------------------------------------------------------- 路由

@app.get("/health")
def health() -> dict:
    """健康检查。部署到容器/K8s 时探针就靠它。"""
    store: VectorStore = state.get("store")
    return {
        "status": "ok",
        "chunks": len(store) if store else 0,
        "llm_model": settings.llm_model,
        "session_backend": state["sessions"].backend if "sessions" in state else "n/a",
    }


@app.post("/api/search")
def api_search(req: SearchRequest) -> dict:
    """只检索不生成，用于调试召回效果。"""
    rag: RAGService = state["rag"]
    hits = rag.retrieve(req.query, top_k=req.top_k)
    return {"query": req.query, "hits": [h.to_dict() for h in hits]}


@app.post("/api/chat")
def api_chat(req: ChatRequest) -> dict:
    """RAG 问答：固定「先检索再生成」流程。"""
    return state["rag"].answer(req.question, session_id=req.session_id)


@app.post("/api/agent")
def api_agent(req: ChatRequest) -> dict:
    """Agent 问答：由模型决定要不要调工具、调哪个。"""
    return state["agent"].run(req.question, session_id=req.session_id)


@app.post("/api/chat/stream")
def api_chat_stream(req: ChatRequest) -> StreamingResponse:
    """SSE 流式问答。

    SSE 的报文格式很简单：每条消息以 `data:` 开头，以空行结束。
    前端用 fetch + ReadableStream 逐段读取即可（见下方首页的示例代码）。
    """

    def event_source():
        for event in state["rag"].answer_stream(req.question, session_id=req.session_id):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/api/session/{session_id}")
def api_clear_session(session_id: str) -> dict:
    state["sessions"].clear(session_id)
    return {"cleared": session_id}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """一个极简的网页版调试界面，不写前端也能直接验证效果。"""
    return INDEX_HTML


INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>知识库问答 Agent</title>
<style>
  body { font-family: -apple-system, "Microsoft YaHei", sans-serif; max-width: 760px;
         margin: 40px auto; padding: 0 16px; color: #222; }
  h1 { font-size: 20px; }
  #log { border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; min-height: 220px;
         white-space: pre-wrap; line-height: 1.7; font-size: 14px; background: #fafafa; }
  .q { color: #1d4ed8; font-weight: 600; }
  .a { color: #111; }
  .c { color: #6b7280; font-size: 12px; }
  .row { display: flex; gap: 8px; margin-top: 12px; }
  input { flex: 1; padding: 10px; border: 1px solid #d1d5db; border-radius: 8px; font-size: 14px; }
  button { padding: 10px 18px; border: 0; border-radius: 8px; background: #1d4ed8;
           color: #fff; font-size: 14px; cursor: pointer; }
</style>
</head>
<body>
<h1>知识库问答 Agent</h1>
<div id="log">输入问题开始，答案会逐字流式返回，并在下方列出引用来源。</div>
<div class="row">
  <input id="q" placeholder="例如：年假有几天？" autofocus>
  <button onclick="ask()">提问</button>
</div>
<script>
const log = document.getElementById('log');
const input = document.getElementById('q');

function push(cls, text) {
  const div = document.createElement('div');
  div.className = cls;
  div.textContent = text;
  log.appendChild(div);
  return div;
}

async function ask() {
  const question = input.value.trim();
  if (!question) return;
  input.value = '';
  push('q', '问：' + question);
  const answer = push('a', '答：');

  const resp = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({question, session_id: 'web'})
  });

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const {value, done} = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, {stream: true});
    // SSE 消息之间用空行分隔
    const parts = buffer.split('\\n\\n');
    buffer = parts.pop();
    for (const part of parts) {
      if (!part.startsWith('data: ')) continue;
      const payload = part.slice(6);
      if (payload === '[DONE]') continue;
      const event = JSON.parse(payload);
      if (event.type === 'delta') answer.textContent += event.text;
      if (event.type === 'citations' && event.items.length) {
        push('c', '引用：' + event.items.map(i => `[${i.index}] ${i.source}`).join('  '));
      }
    }
  }
}

input.addEventListener('keydown', e => { if (e.key === 'Enter') ask(); });
</script>
</body>
</html>
"""
