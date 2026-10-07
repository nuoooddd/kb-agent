"""会话记忆：让多轮对话能接上上下文。

【为什么不能直接把所有历史都塞给模型】
1. 上下文有长度上限，塞不下；
2. 越长越贵，而且模型对中间部分的信息注意力会下降；
3. 大部分场景里，「最近几轮」才真正影响当前回答。

所以主流做法是【滑动窗口】：只保留最近 N 轮，更早的丢掉或压缩成摘要。
本项目实现滑动窗口；摘要压缩尚未实现，见 README「已知不足」。

【为什么要有 Redis 这一层】
先用内存 dict 也能跑，但：
- 进程重启历史就没了；
- 起多个实例时各存各的，用户被负载均衡打到另一台就「失忆」了；
- 没有 TTL，长期运行会内存泄漏。
所以生产要用 Redis。这里做成「没配 REDIS_URL 就自动退化为内存版」，
不装 Redis 也能跑起来。
"""

from __future__ import annotations

import json
import time


class SessionStore:
    def __init__(
        self,
        redis_url: str = "",
        max_turns: int = 8,
        ttl: int = 3600,
    ) -> None:
        self.max_turns = max_turns
        self.ttl = ttl
        self.backend = "memory"
        self._memory: dict[str, list[dict]] = {}
        self._redis = None

        if redis_url:
            try:
                import redis  # 没装 redis 包时会 ImportError

                client = redis.Redis.from_url(redis_url, decode_responses=True)
                client.ping()
                self._redis = client
                self.backend = "redis"
            except Exception as exc:  # noqa: BLE001
                # 连不上就降级，绝不让会话功能把整个服务拖挂
                print(f"[session] Redis 不可用({exc})，降级为内存存储")

    # ------------------------------------------------------------------ key

    def _key(self, session_id: str) -> str:
        return f"kb-agent:session:{session_id}"

    # ------------------------------------------------------------------ 读

    def get(self, session_id: str) -> list[dict]:
        """取回该会话的历史消息（已经是裁剪过的）。"""
        if self._redis is not None:
            raw = self._redis.get(self._key(session_id))
            if not raw:
                return []
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return []
        return list(self._memory.get(session_id, []))

    # ------------------------------------------------------------------ 写

    def append(self, session_id: str, role: str, content: str) -> None:
        """追加一条消息，并按滑动窗口裁剪。"""
        history = self.get(session_id)
        history.append({"role": role, "content": content, "ts": int(time.time())})
        # 一轮 = 一问一答，所以窗口上限是 2 倍
        history = history[-self.max_turns * 2 :]

        if self._redis is not None:
            # 用流水线把「写值 + 设过期」打包，避免中间状态
            pipe = self._redis.pipeline()
            pipe.set(self._key(session_id), json.dumps(history, ensure_ascii=False))
            pipe.expire(self._key(session_id), self.ttl)
            pipe.execute()
        else:
            self._memory[session_id] = history

    def clear(self, session_id: str) -> None:
        if self._redis is not None:
            self._redis.delete(self._key(session_id))
        else:
            self._memory.pop(session_id, None)

    def to_llm_messages(self, session_id: str) -> list[dict]:
        """剥掉 ts 等内部字段，只留模型认识的 role/content。"""
        return [
            {"role": m["role"], "content": m["content"]}
            for m in self.get(session_id)
            if m.get("role") in {"user", "assistant"} and m.get("content")
        ]
