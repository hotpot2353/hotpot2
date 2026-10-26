# 技术面Agent：行情时序纯代码解析 + LLM 叙述两段
# LLM 边界：均线/量比/动量/支撑压力/极值全部纯代码计算，LLM 只写
#   趋势判断与资金结论两段叙述；关键价格区间与风险点由代码直接产出，数字不经 LLM
# 数据边界：quotes 仅 close/volume 两列，无真实资金流——资金结论为量价代理推断（提示词已声明）
# 极值处理：大幅波动日只标记不剔除（项目铁律），风险点据此提示
import json

import requests

from agents.orchestrator.state import AgentState
from core.config import get_settings, load_yaml
from core.logger import get_logger
from core.prompt_loader import load_prompt
from core.trace import record_tokens
from llm.chat_client import get_chat_client

logger = get_logger(__name__)

# 技术参数文件：均线窗口/量比阈值/支撑压力回看期等（改阈值不改代码）
_RULES_FILE = "rules/technical_rules.yaml"


def run(state: AgentState) -> dict:
    # 技术面入口：算特征快照 → 代码产支撑压力与风险点 → LLM 两段叙述 → 契约组装
    quotes = state.raw_data.get("quotes") or {}
    dates = quotes.get("dates") or []
    closes = quotes.get("close") or []
    volumes = quotes.get("volume") or []

    # 三序列必须等长且至少 2 期（算日涨跌幅的底线）；池外标的行情=0 走此降级
    if len(closes) < 2 or not (len(closes) == len(volumes) == len(dates)):
        logger.warning("[technical] 行情数据不足 len=%s 降级输出", len(closes))
        return _empty("行情数据不足，本报告不含技术面分析")

    cfg = load_yaml(_RULES_FILE)
    code = str(state.intent.get("stock_code", ""))
    feats = _compute(dates, closes, volumes, cfg, code)
    risks = _risk_points(feats, cfg)
    trend_text, money_text = _narrate(state, feats, cfg)

    logger.info("[technical] %s 形态=%s 量比=%.2f 极值日=%d",
                code, feats["alignment"], feats["vol_ratio"], len(feats["extremes"]))
    return {
        "趋势判断": trend_text,
        "资金结论": money_text,
        "关键价格区间": {"支撑": feats["support"], "压力": feats["pressure"]},
        # 均线形态透传：风控矛盾检测要结构化字段而非从叙述文本反解
        "均线形态": feats["alignment"],
        "风险点": risks,
        # evidence_ref 契约同 fundamental：本节点消费结构化行情，无原文引用
        "evidence_ref": [],
    }


def _empty(reason: str) -> dict:
    # 降级输出：契约形状恒定，缺的数字显式置空并附边界说明（禁止装作有结论）
    return {
        "趋势判断": reason,
        "资金结论": "",
        "关键价格区间": {"支撑": None, "压力": None},
        "均线形态": "",
        "风险点": [],
        "evidence_ref": [],
    }


def _compute(dates: list, closes: list, volumes: list, cfg: dict, code: str) -> dict:
    # 技术特征快照：全部纯代码计算，数字可复现（LLM 只消费不产生）
    feats = {
        "last_close": closes[-1],
        "ma": {},
        "alignment": "均线纠缠",
        "above_key_ma": None,
        "vol_ratio": 1.0,
        "last_change": closes[-1] / closes[-2] - 1,
        "momentum": {},
        "support": None,
        "pressure": None,
        "range_pos": None,
        "extremes": [],
    }

    # 均线：窗口期数不足则跳过该线（数据边界显式降级，不硬算）
    for w in cfg["ma_windows"]:
        if len(closes) >= w:
            feats["ma"][f"MA{w}"] = sum(closes[-w:]) / w
    # 排列判定取最短三根（cfg 升序前三个）：MA5>MA10>MA20 多头，反之空头，否则纠缠
    short3 = sorted(cfg["ma_windows"])[:3]
    mas = [feats["ma"][f"MA{w}"] for w in short3 if f"MA{w}" in feats["ma"]]
    if len(mas) == 3:
        if mas[0] > mas[1] > mas[2]:
            feats["alignment"] = "多头排列"
        elif mas[0] < mas[1] < mas[2]:
            feats["alignment"] = "空头排列"
    if "MA20" in feats["ma"]:
        feats["above_key_ma"] = closes[-1] > feats["ma"]["MA20"]

    # 量比：最新成交量相对前 N 日均量（不含当日）
    vw = int(cfg["vol_ma_window"])
    if len(volumes) > vw:
        base = sum(volumes[-vw - 1:-1]) / vw
        if base > 0:
            feats["vol_ratio"] = volumes[-1] / base

    # 动量：近 N 日涨跌幅
    for w in cfg["momentum_windows"]:
        if len(closes) > w:
            feats["momentum"][f"{w}日"] = closes[-1] / closes[-1 - w] - 1

    # 支撑压力：近 N 日收盘最低/最高（前期高低点是最近邻的支撑/压力参考）
    window = closes[-int(cfg["range_window"]):]
    feats["support"] = min(window)
    feats["pressure"] = max(window)

    # 全区间高低分位：当前价在最近 120 交易日区间的位置
    lo, hi = min(closes), max(closes)
    if hi > lo:
        feats["range_pos"] = (closes[-1] - lo) / (hi - lo)

    # 大幅波动日：|日涨跌幅| ≥ 阈值（主板/创科板不同涨跌停限制），只标记不剔除
    th_star = float(cfg["extreme_move_pct_star"])
    th = th_star if code.startswith(("300", "688")) else float(cfg["extreme_move_pct_main"])
    for i in range(1, len(closes)):
        pct = closes[i] / closes[i - 1] - 1
        if abs(pct) >= th:
            feats["extremes"].append((dates[i], round(pct, 4)))
    feats["extremes"] = feats["extremes"][-3:]
    return feats


def _risk_points(feats: dict, cfg: dict) -> list:
    # 风险点判定：零 LLM 规则清单；无命中返回空列表（报告层自行决定是否呈现）
    risks = []
    if feats["vol_ratio"] >= float(cfg["vol_surge_ratio"]) and feats["last_change"] < 0:
        risks.append("放量下跌，短期抛压明显")
    if feats["above_key_ma"] is False:
        risks.append("收盘价跌破20日均线")
    w_max = max(cfg["momentum_windows"])
    m_max = feats["momentum"].get(f"{w_max}日")
    if m_max is not None and m_max <= float(cfg["momentum_drawdown_threshold"]):
        risks.append(f"近{w_max}日累计跌幅超15%")
    if feats["extremes"]:
        d, p = feats["extremes"][-1]
        risks.append(f"近期存在大幅波动日（{d} {p:+.1%}），波动风险抬升")
    return risks


def _snapshot_lines(feats: dict) -> list:
    # 特征快照文本：喂给 LLM 的全部数字来源，一行一个特征
    lines = []
    if feats["ma"]:
        lines.append("均线 " + " ".join(f"{k}={v:.2f}" for k, v in feats["ma"].items()))
    lines.append(f"均线形态 {feats['alignment']}")
    if feats["above_key_ma"] is not None:
        pos = "上方" if feats["above_key_ma"] else "下方"
        lines.append(f"收盘 {feats['last_close']} 位于 MA20 {pos}")
    lines.append(f"最新收盘 {feats['last_close']}，当日涨跌幅 {feats['last_change']:+.2%}")
    lines.append(f"量比 {feats['vol_ratio']:.2f}（最新成交量相对前5日均量）")
    for w, m in feats["momentum"].items():
        lines.append(f"近{w}涨跌幅 {m:+.2%}")
    if feats["range_pos"] is not None:
        lines.append(f"价格处于近120交易日高低区间的 {feats['range_pos']:.0%} 分位")
    if feats["extremes"]:
        es = "；".join(f"{d} {p:+.1%}" for d, p in feats["extremes"])
        lines.append(f"大幅波动日样本：{es}")
    return lines


def _narrate(state: AgentState, feats: dict, cfg: dict) -> tuple:
    # LLM 叙述两段（趋势判断/资金结论）；失败回落规则拼接，链路不炸
    prompt = load_prompt("technical_agent")
    user_content = (prompt.user
                    .replace("{stock_name}", str(state.intent.get("stock_name", "")))
                    .replace("{stock_code}", str(state.intent.get("stock_code", "")))
                    .replace("{snapshot}", "\n".join(_snapshot_lines(feats))))
    client = get_chat_client(model=get_settings().ark.llm_model)
    messages = [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": user_content},
    ]
    try:
        usage = {}
        text = client.chat(messages, temperature=0.2, max_tokens=600, thinking="disabled", usage=usage)
        record_tokens(state, "technical", usage)
        data = _extract_json(text)
    except (ValueError, requests.RequestException) as exc:
        # ValueError 覆盖 JSONDecodeError；网络异常同属边界可恢复故障
        logger.warning("技术面 LLM 叙述失败，走规则兜底: %s", exc)
        state.errors.append({"type": "technical_llm_error", "detail": str(exc)})
        return _fallback(feats, cfg)

    # LLM 键漂移兼容：中文键优先，英文键回落
    trend = str(data.get("趋势判断") or data.get("trend") or "").strip()
    money = str(data.get("资金结论") or data.get("money") or "").strip()
    if not trend:
        logger.warning("技术面 LLM 输出缺趋势判断，走规则兜底")
        state.errors.append({"type": "technical_llm_error", "detail": "missing trend key"})
        return _fallback(feats, cfg)
    return trend, money


def _fallback(feats: dict, cfg: dict) -> tuple:
    # 规则兜底：LLM 不可用时报告仍有数据支撑的技术结论（数字全部来自特征快照）
    trend = (f"最新收盘 {feats['last_close']}，均线{feats['alignment']}，"
             f"当日涨跌 {feats['last_change']:+.2%}。")
    if feats["above_key_ma"] is True:
        trend += "价格站上20日均线。"
    elif feats["above_key_ma"] is False:
        trend += "价格位于20日均线下方。"
    if feats["vol_ratio"] >= float(cfg["vol_surge_ratio"]):
        money = f"量比 {feats['vol_ratio']:.2f}，明显放量，量价活跃度提升。"
    elif feats["vol_ratio"] <= float(cfg["vol_shrink_ratio"]):
        money = f"量比 {feats['vol_ratio']:.2f}，明显缩量，交投清淡。"
    else:
        money = f"量比 {feats['vol_ratio']:.2f}，量能平稳。"
    return trend, money


def _extract_json(text: str) -> dict:
    # 剥代码围栏 + 首尾花括号截取，防 LLM 输出夹带说明文字
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`").lstrip("json").strip()
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("输出不含 JSON 对象")
    return json.loads(t[start:end + 1])
