"""大模型调用封装（OpenAI 兼容接口）。

只要服务商提供 OpenAI 兼容协议（智谱 GLM、DeepSeek、通义、Kimi、本地 vLLM 等），
换 base_url + model 名就能切换，代码一行不用改。这也是用 openai SDK
而不是各家私有 SDK 的原因——把「换模型」的成本降到最低。

对外提供两件事：
    chat(messages, tools=...)  一次性拿到完整回答；传 tools 时模型会自己决定要不要调用工具
                               （这就是 Agent 的基础，agent.py 走的是这条路）
    chat_stream()              逐字吐出增量（用户体验好，首 token 延迟低）
"""

from __future__ import annotations

from collections.abc import Iterator

from app.config import settings

_client = None


def get_client():
    global _client
    if _client is None:
        if not settings.llm_api_key:
            raise RuntimeError(
                "没有找到 LLM_API_KEY。请在项目根目录创建 .env，"
                "参考 .env.example 填入你的密钥。"
            )
        from openai import OpenAI

        _client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    return _client


def _dump_tool_calls(tool_calls) -> list[dict]:
    """把 SDK 对象转成纯 dict，方便塞回 messages 和输出 JSON。"""
    if not tool_calls:
        return []
    return [
        {
            "id": call.id,
            "type": "function",
            "function": {
                "name": call.function.name,
                "arguments": call.function.arguments,
            },
        }
        for call in tool_calls
    ]


def chat(
    messages: list[dict],
    tools: list[dict] | None = None,
    temperature: float | None = None,
) -> dict:
    """一次完整对话。返回 {"content": str, "tool_calls": list}。"""
    kwargs: dict = {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": settings.temperature if temperature is None else temperature,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    resp = get_client().chat.completions.create(**kwargs)
    message = resp.choices[0].message
    return {
        "content": message.content or "",
        "tool_calls": _dump_tool_calls(getattr(message, "tool_calls", None)),
    }


def chat_stream(messages: list[dict], temperature: float | None = None) -> Iterator[str]:
    """流式对话：每 yield 一小段新增文本。

    流式不是「更快」，而是「更早看到第一个字」。总耗时几乎不变，
    但用户感知的等待时间从 5 秒变成 0.5 秒——这就是它的价值。
    """
    stream = get_client().chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=settings.temperature if temperature is None else temperature,
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        piece = getattr(delta, "content", None)
        if piece:
            yield piece
