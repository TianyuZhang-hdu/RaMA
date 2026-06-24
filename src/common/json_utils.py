"""
JSON 解析相关工具函数
"""

import json


def extract_first_json(text: str) -> str:
    """
    从文本中提取第一个完整的 JSON 对象（处理括号匹配）
    
    Args:
        text: 包含 JSON 的文本
    
    Returns:
        提取的 JSON 字符串，如果未找到则返回 None
    """
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    for i, c in enumerate(text[start:], start):
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None


def parse_llm_response(resp: str) -> dict:
    """
    解析 LLM 返回的响应，提取 JSON 结果
    
    Args:
        resp: LLM 返回的原始响应
    
    Returns:
        解析后的字典，失败时返回错误结构
    """
    if not resp:
        print(f"  [ERROR] LLM返回空")
        return {"scores": {}, "decision": "ERROR", "fp_points": [], "fn_points": []}
    
    print(f"  [DEBUG] 返回长度: {len(resp)}")
    
    resp_clean = resp.strip()
    if resp_clean.startswith("```"):
        resp_clean = resp_clean.split("\n", 1)[1]
        resp_clean = resp_clean.rsplit("```", 1)[0]
    
    json_str = extract_first_json(resp_clean)
    if json_str is None:
        print(f"  [ERROR] 未找到 JSON")
        print(f"  [ERROR] 原始响应: {resp[:500]}")
        return {"scores": {}, "decision": "ERROR", "fp_points": [], "fn_points": []}
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"  [ERROR] JSON解析失败: {e}")
        print(f"  [ERROR] 提取的内容: {json_str[:500]}")
        print(f"  [ERROR] 原始响应: {resp[:500]}")
        return {"scores": {}, "decision": "ERROR", "fp_points": [], "fn_points": []}

