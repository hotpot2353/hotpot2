# 意图识别：query → 结构化 JSON（LLM 边界内：单点推理，不做任务拆分）
# 流程：load_prompt 装载三段式提示词 → str.replace 填充占位符 → chat 客户端
#       （thinking=disabled 省思考开销）→ 提取 JSON → 六字段规范化补默认
import json

from agents.orchestrator.state import AgentState
from core.config import get_settings
from core.prompt_loader import load_prompt
from core.trace import record_tokens
from llm.chat_client import get_chat_client

# 六字段缺省值（提示词约定：缺省补默认，不输出 null）
_DEFAULT_YEARS = [2024]
# 缺省维度用 list：有序容器，直接整份给下游用，下游渲染研报章节顺序稳定可依赖
_DEFAULT_DIMENSIONS = ["fundamental", "technical", "sentiment"]
# 合法维度用 set：无序不重复容器，只做成员判断（in 查找一步直达）
_VALID_DIMENSIONS = {"fundamental", "technical", "sentiment"}


def run(query: str, state: AgentState) -> dict:
    # 意图识别入口：query 为用户原始问题，返回规范化后的六字段 intent dict
    # state 仅用于 trace 记账（token 回填），不影响识别逻辑
    prompt = load_prompt("orchestrator_intent")
    # 占位符用 replace 填充：system 段含 JSON 示例花括号，str.format 会 KeyError
    user_content = prompt.user.replace("{query}", query)

    # get_chat_client 按参数缓存单例；model 从配置现取显式 Model ID 防别名漂移
    client = get_chat_client(model=get_settings().ark.llm_model)
    # OpenAI 消息格式：system 在前定角色与输出契约，user 在后给任务输入，顺序有语义
    messages = [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": user_content},
    ]
    # temperature=0 求判定可复现；thinking=disabled 关闭思考模型深度推理
    usage = {}
    text = client.chat(messages, temperature=0.0, thinking="disabled", usage=usage)
    record_tokens(state, "intent", usage)
    # 格式层解析：剥围栏 + 取首尾花括号，容忍 LLM 在 JSON 前后的絮叨文字
    data = _extract_json(text)
    # 契约层规范化：类型纠正 + 白名单过滤 + 缺省补默认，保证下游拿到恒定形状
    return _normalize(data, query)


def _extract_json(text: str) -> dict:
    # 从 LLM 回复中提取 JSON：剥代码围栏后取首尾花括号之间解析
    text = text.strip()
    if text.startswith("```"):
        # 剥掉 ```json ... ``` 围栏，保留围栏内正文
        text = text.strip("`").lstrip("json").strip()
    # find 取第一个 { 、rfind 取最后一个 }——JSON 前后无论 LLM 说了什么，主体必在两者之间
    start, end = text.find("{"), text.rfind("}")
    # 无花括号或位置颠倒（空/残缺）→ 炸在源头；异常截前 200 字符防撑爆日志
    if start == -1 or end <= start:
        raise ValueError(f"意图识别回复无 JSON 结构: {text[:200]}")
    # 切片下标含尾需 end+1（切片规则：不含尾）；解析失败 json.loads 自会抛错同样炸在源头
    return json.loads(text[start:end + 1])


def _normalize(data: dict, query: str) -> dict:
    # 六字段规范化：类型纠正 + 非法值回落默认，保证下游拿到的形状恒定
    # str() 转 LLM 可能给的数字形态；strip() 去空格；缺字段给空串防 None 链
    stock_code = str(data.get("stock_code", "")).strip()
    # 唯一 raise 的字段：代码错则全链路查错公司，没有合理默认，必须炸
    if not stock_code.isdigit() or len(stock_code) != 6:
        raise ValueError(
            f"未能从问题中识别出 6 位股票代码（query={query!r}）；"
            "本系统仅支持单只股票研判，请在问题中写明股票代码或名称，"
            "如：对招商银行 2024 年做完整研判")

    # or 兜底：LLM 给 null/[]/缺失（均为 falsy）一律回落默认，一行覆盖四种坏形态
    years = data.get("years") or _DEFAULT_YEARS
    # int() 一石二鸟：类型纠正（LLM 可能给 2024.0）+ 只留库内年份（DB 仅灌 2022-2024）
    years = [int(y) for y in years if int(y) in (2022, 2023, 2024)]
    # 全被过滤（如 LLM 给了 2020 等库外年份）→ 回落最新报告期
    if not years:
        years = list(_DEFAULT_YEARS)

    # 白名单过滤：d in set 是哈希直达；LLM 幻觉出的池外维度（如 "valuation"）直接剔除
    dimensions = [d for d in (data.get("dimensions") or []) if d in _VALID_DIMENSIONS]
    # 全被剔除 → 回落三维全开
    if not dimensions:
        dimensions = list(_DEFAULT_DIMENSIONS)

    # 成文形态：只认 report/summary 两个合法值，LLM 自创值回落 report
    output = data.get("output") if data.get("output") in ("report", "summary") else "report"

    # 形状恒定的六字段：键全、类型定、无 null，下游可盲信不必写任何防御
    return {
        "stock_code": stock_code,
        "stock_name": str(data.get("stock_name", "")).strip(),
        "years": years,
        "dimensions": dimensions,
        "output": output,
        "extra": str(data.get("extra", "")).strip(),
    }
