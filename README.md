# kb-agent

基于 RAG（检索增强生成）的知识库问答服务。把一批文档灌进去，就能问出**带引用出处**的答案；
模型也可以自主决定调用哪个工具来取数据。

## 核心特性

- **混合检索**：向量召回（语义）+ BM25 召回（关键词）双路并行，各自归一化后加权融合。
  用于解决「专有名词 / 编号 / 缩写用纯语义向量搜不到」的问题。
- **引用溯源**：回答里的每个结论都带 `[编号]`，可回溯到原始文档片段。
- **拒答机制**：检索最高分低于阈值时直接短路返回、不调用大模型，避免无依据的编造。
- **检索层零第三方依赖**：递归切分与检索（余弦相似度 / BM25 / 融合）全部用 Python
  标准库实现，不依赖向量数据库，可以完全离线运行与验证。
- **Function Calling**：模型自主决定调用哪个工具，支持多轮工具循环。
- **会话记忆**：滑动窗口；配置了 Redis 就用 Redis，没配则自动降级到进程内存。
- **可量化评估**：内置评估脚本，输出 Recall@1/3/5、MRR 与拒答准确率。

## 架构

```
用户提问
   │
   ├─ POST /api/search        只检索
   ├─ POST /api/chat          固定流程：先检索 → 再生成（RAG）
   ├─ POST /api/chat/stream   同上，SSE 逐字流式返回
   └─ POST /api/agent         模型自主决定调哪个工具（Agent）
                    │
        ┌───────────┴───────────┐
        │  混合检索              │  向量召回（语义） + BM25 召回（关键词）
        │  → 归一化 → 加权融合   │
        └───────────┬───────────┘
                    │
        ┌───────────┴───────────┐
        │  向量存储（index.json）│  余弦相似度，标准库实现
        └───────────┬───────────┘
                    │
        ┌───────────┴───────────┐
        │  会话记忆              │  滑动窗口；有 Redis 用 Redis，没有自动降级内存
        └───────────────────────┘
```

## 快速开始

```bash
# 0. 先看效果：下面三条不需要装依赖、不需要密钥
python tests/test_core.py          # 核心逻辑单元测试
python -m app.ingest --dry-run     # 看文档被切成什么样
python -m app.ingest --no-embed && python -m app.eval --keyword-only   # 纯关键词检索的召回率

# 1. 装依赖
pip install -r requirements.txt

# 2. 配密钥
cp .env.example .env               # Windows: copy .env.example .env
#   编辑 .env，填 LLM_API_KEY（LLM_BASE_URL / LLM_MODEL 要和服务商匹配）

# 3. 把 data/ 里的文档向量化入库（会覆盖第 0 步的 --no-embed 索引）
python -m app.ingest

# 4. 起服务
uvicorn app.main:app --reload --port 8000

# 5. 评估检索效果
python -m app.eval
```

服务起来后：

- 网页调试界面：<http://127.0.0.1:8000/>
- 交互式接口文档：<http://127.0.0.1:8000/docs>

## 接口示例

```bash
# 只检索，看召回效果
curl -X POST http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query":"RAG 是谁提出的","top_k":3}'

# RAG 问答
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"思维链提示是谁提出的？","session_id":"u1"}'

# Agent 问答（会返回它调用了哪些工具）
curl -X POST http://127.0.0.1:8000/api/agent \
  -H "Content-Type: application/json" \
  -d '{"question":"现在几点？顺便说说 Reflexion 是什么","session_id":"u1"}'

# 流式
curl -N -X POST http://127.0.0.1:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"自我一致性是怎么实现的","session_id":"u1"}'
```

## 目录结构

| 文件 | 职责 |
| --- | --- |
| `app/chunker.py` | 递归切分 + overlap，纯标准库 |
| `app/vector_store.py` | 余弦相似度 / BM25 / 混合检索，纯标准库 |
| `app/rag.py` | 检索 → 拼 Prompt → 生成 → 引用 + 拒答 |
| `app/agent.py` | Function Calling 多轮工具循环 |
| `app/embeddings.py` | 向量化封装（批量 + 顺序修正） |
| `app/llm.py` | 大模型调用（同步 / 流式 / 带工具） |
| `app/eval.py` | Recall@K / MRR / 拒答准确率 |
| `app/session.py` | 会话记忆（滑动窗口 + Redis 可选） |
| `app/ingest.py` | 离线入库脚本，支持 `--dry-run` / `--no-embed` |
| `app/main.py` | FastAPI 路由 + 内置调试页面 |
| `tests/test_core.py` | 单元测试，不装 pytest 也能跑 |

## 实测结果

测试环境：13 篇演示文档（30,173 字）→ 62 个 chunk（`chunk_size=600 / overlap=100`）；
自建 50 条评估集（25 条字面题 + 15 条同义改写 + 10 条不可答题）。

```
python tests/test_core.py         →  全部用例通过
python -m app.ingest --dry-run    →  13 篇 → 62 个 chunk，平均 487 字
python -m app.ingest              →  索引 62 条，embedding 维度 1024
python -m app.eval                →  Recall@1 92.5% / @3 100% / @5 100%，MRR 0.9583
                                     拒答准确率 10/10（阈值 0.52）
uvicorn + POST /api/chat          →  返回带引用编号的回答，grounded = true
```

| 指标 | 数值 | 含义 |
| --- | --- | --- |
| Recall@1 | 37/40 = **92.5%** | 可答题里正确答案排第一的比例 |
| Recall@3 | 40/40 = **100%** | 排进前三的比例 |
| Recall@5 | 40/40 = **100%** | 排进前五的比例 |
| MRR | **0.9583** | 平均倒数排名，越接近 1 越好 |
| 拒答准确率 | **10/10** | 10 条不可答题全部被阈值正确挡下 |

> **未进 Top-1 的 3 条**全部是同义改写题、且答案为并列项（RAG 提出者、Reflexion 的三个
> 角色、ToT 的三个标签），掉到第 2~3 名。直接复用原文措辞的字面题几乎都稳在 Top-1——
> 这组对照说明：单看字面题会高估检索能力，改写题才是真实的区分度来源。

> **关于拒答**：判定看的是原始余弦相似度，不是归一化后的融合分（后者最高恒为 1.0，
> 拿它比阈值永远成立）。阈值 0.52 的标定过程与局限见「已知不足」第 1 条。

## 演示语料

`data/` 下的 13 篇 `zh-*.md` 取自开源文档
[Prompt Engineering Guide 中文版](https://www.promptingguide.ai/zh)，仅用于演示。

换成自己的文档：把文件丢进 `data/` → `python -m app.ingest` 重建索引 →
按自己的语料重写 `data/eval_set.json` → `python -m app.eval` 得到自己的指标。
评估集格式见 `data/eval_set.json`：`must_contain` 是判分时要求同时出现在**同一个 chunk**
里的字符串，所以选词要足够专用，否则会把「答错」误判成「命中」；
`tools/check_eval_set.py` 可以批量检查这一点。

## 已知不足

1. **拒答阈值靠标定，泛化性有限**：判定改用原始余弦相似度后，阈值才真正生效
   （之前用归一化融合分，最高恒为 1.0，一次都触发不了）。但可答题与不可答题的
   余弦区间只差约 0.03，`0.52` 是在这 50 条评估集上标定的：集内 10/10 全挡对、
   误伤 1 条可答题；换到集外措辞更短的问法（如「RAG 是什么时候提出的」，余弦 0.487）
   就会被误拒。更稳的做法是两段式：先阈值粗筛，再由模型判定能否回答。
2. **索引全量重算**：`ingest` 每次都重建整个索引，文档多了需要做增量更新和去重。
3. **向量检索是自研的**：上万条以上要换 Qdrant / Chroma，JSON 反序列化扛不住。
4. **BM25 分词是字符二元组**：中文长文档应该换 jieba 之类的真实分词器。
5. **没有 rerank**：混合检索是简单加权，生产里常用 cross-encoder 对 Top-N 精排。
6. **会话历史没有摘要压缩**：超长对话会撑爆上下文窗口，现在只是粗暴截断。
7. **同步调用占用线程池**：并发高时应该换异步 HTTP 客户端，全程 await。
8. **无鉴权**：公开部署前必须加 API Key / JWT 和限流。

## Roadmap

1. ~~修复归一化导致的阈值失效，让拒答判定真正可比~~ —— 已完成，改用原始余弦判定。
2. ~~换成真实语料，评估集扩到 50 条（含同义改写题与不可答题）~~ —— 已完成，13 篇 / 62 chunk。
3. 加第三个工具：Text2SQL 查只读业务库。
4. 加 rerank：对混合检索的 Top-20 做 cross-encoder 精排再取 Top-K。
5. 换 Qdrant，并处理元数据过滤与增量更新。
6. Docker 化，`docker compose up` 一键起。
