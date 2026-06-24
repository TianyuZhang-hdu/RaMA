import base64
import os
import time
from typing import Any, Optional

from openai import OpenAI
import config as _config

def _img_to_base64(path):
    ext = os.path.splitext(path)[1].lower()
    mt = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
    }.get(ext, "image/png")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mt};base64,{b64}"

def send_generate_request(messages, server_url=None, model="", api_key=None,
                          max_tokens=4096, max_retries=3, enable_thinking=None):
    """
    发送 LLM 请求。当 enable_thinking=True 时使用流式调用 + 深度思考模式。
    """
    if enable_thinking is None:
        enable_thinking = getattr(_config, "ENABLE_THINKING", False)

    processed = []
    for m in messages:
        mm = m.copy()
        if m.get("role") == "user" and isinstance(m.get("content"), list):
            newc = []
            for c in m["content"]:
                if isinstance(c, dict) and c.get("type") == "image":
                    url = _img_to_base64(c["image"])
                    newc.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})
                else:
                    newc.append(c)
            mm["content"] = newc
        processed.append(mm)

    client = OpenAI(api_key=api_key, base_url=server_url)

    if enable_thinking:
        # 深度思考模式：流式调用，收集 content（忽略 reasoning_content）
        stream = client.chat.completions.create(
            model=model,
            messages=processed,
            max_tokens=max_tokens,
            n=1,
            stream=True,
            extra_body={"enable_thinking": True},
        )
        answer_content = ""
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            # 只收集最终回复内容，跳过思考过程
            if hasattr(delta, "content") and delta.content:
                answer_content += delta.content
        return answer_content if answer_content else None
    else:
        # 普通模式：非流式调用
        resp = client.chat.completions.create(
            model=model, messages=processed,
            max_tokens=max_tokens, n=1
        )
        if resp.choices:
            return resp.choices[0].message.content
        return None

