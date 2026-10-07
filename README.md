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
  -d '{"query":"年假有几天","top_k":3}'

# RAG 问答
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"年假怎么算？","session_id":"u1"}'

# Agent 问答（会返回它调用了哪些工具）
curl -X POST http://127.0.0.1:8000/api/agent \
  -H "Content-Type: application/json" \
  -d '{"question":"今天几号？顺便说说报销标准","session_id":"u1"}'

# 流式
curl -N -X POST http://127.0.0.1:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"报销流程是什么","session_id":"u1"}'
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

本机实测（2026-10）：

```
python tests/test_core.py         →  全部用例通过
python -m app.ingest --dry-run    →  904 字 → 2 个 chunk
python -m app.ingest              →  索引 2 条，embedding 维度 1024
python -m app.eval                →  Recall@1/3/5 = 100%，拒答准确率 0/3
uvicorn + POST /api/chat          →  返回带引用编号的回答，grounded = true
```

> **关于上面这组指标**：当前 `data/` 里只有 1 篇 904 字的示例文档（2 个 chunk），
> 评估集也是照着该文档原文出的、没有干扰项，所以召回率**不具参考价值**——它只说明流程跑通了。
> 反过来，拒答准确率 0/3 是真实存在的短板：**min-max 归一化会把最高分硬拉成 1.0**，
> 导致阈值判定失效，任何问题都能拿到满分。要修复需要改用可直接比较的分数
> （如保留余弦原始值），并扩充语料与评估集后才能验证。
>
> 这两条正是当前的已知问题，见下方「已知不足」第 1 条。

## 已知不足

1. **拒答阈值形同虚设**：混合检索的分数做了 min-max 归一化，只保序、丢掉了绝对差距，
   最高分恒为 1.0，导致 `SCORE_THRESHOLD` 无法区分「真的相关」和「只是相对最高」。
   需要保留可比原始分数，或引入 rerank 后重新标定阈值。
2. **索引全量重算**：`ingest` 每次都重建整个索引，文档多了需要做增量更新和去重。
3. **向量检索是自研的**：上万条以上要换 Qdrant / Chroma，JSON 反序列化扛不住。
4. **BM25 分词是字符二元组**：中文长文档应该换 jieba 之类的真实分词器。
5. **没有 rerank**：混合检索是简单加权，生产里常用 cross-encoder 对 Top-N 精排。
6. **会话历史没有摘要压缩**：超长对话会撑爆上下文窗口，现在只是粗暴截断。
7. **同步调用占用线程池**：并发高时应该换异步 HTTP 客户端，全程 await。
8. **无鉴权**：公开部署前必须加 API Key / JWT 和限流。

## Roadmap

1. 修复归一化导致的阈值失效，让拒答判定真正可比。
2. 把 `data/` 换成真实语料（几十篇文档），评估集扩到 50 条（含同义改写题与不可答题），
   跑出可信的召回率与拒答准确率。
3. 加第三个工具：Text2SQL 查只读业务库。
4. 加 rerank：对混合检索的 Top-20 做 cross-encoder 精排再取 Top-K。
5. 换 Qdrant，并处理元数据过滤与增量更新。
6. Docker 化，`docker compose up` 一键起。
