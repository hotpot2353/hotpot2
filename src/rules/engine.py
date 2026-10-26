# 财务指标组合研判规则引擎：纯代码方向判定 + 规则匹配，零 LLM（LLM 使用边界）
# 消费方：基本面Agent（组合研判，命中按条件数降序取 top-2）、风控Agent第1层（行业异动硬阈值）
# 业务数值全部来自 configs/rules/analysis_rules.yaml，引擎不写死任何业务数值
from core.config import load_yaml
from core.logger import get_logger

logger = get_logger(__name__)

# 规则文件相对 configs/ 的路径
_RULES_FILE = "rules/analysis_rules.yaml"


class RuleEngine:
    # 规则引擎：yaml 装载一次，classify / match / risk_flags 可反复调用（无状态）
    def __init__(self, rules_file: str = _RULES_FILE):
        raw = load_yaml(rules_file)
        self.thresholds = raw["direction_thresholds"]
        self.indicator_types = raw["indicator_types"]
        self.rules = raw["rules"]
        self._risk = raw["risk_industry_thresholds"]
        self.stable_pp = float(self.thresholds["stable_pp"])
        self.stable_rel = float(self.thresholds["stable_rel"])
        self.large_rel = float(self.thresholds["large_change_rel"])

    def classify(self, series: dict) -> dict:
        # 方向判定：年度序列相邻两期即同比，取最新一期定方向
        # series: {指标键: [v_最早, ..., v_最新]}，至少 2 期
        # 返回 {指标键: {direction, magnitude, persist, latest, values}}（match 的输入）
        snapshot = {}
        for ind, vals in series.items():
            if len(vals) < 2:
                raise ValueError(f"指标 {ind} 序列不足 2 期，无法判定方向")
            vals = [float(v) for v in vals]
            cur, prev = vals[-1], vals[-2]
            rel = self._rel(cur, prev)
            if self.indicator_types.get(ind, "rel") == "pp":
                delta = cur - prev
                direction = "stable" if abs(delta) <= self.stable_pp else ("up" if delta > 0 else "down")
            else:
                direction = "stable" if abs(rel) <= self.stable_rel else ("up" if rel > 0 else "down")
            snapshot[ind] = {
                "direction": direction,
                "magnitude": "large" if abs(rel) >= self.large_rel else "normal",
                "persist": self._persist_count(vals, ind),
                "latest": cur,
                "values": vals,
            }
        return snapshot

    def match(self, snapshot: dict, top_n: int = 2) -> list:
        # 规则匹配：全部条件满足才算命中；多命中按条件数降序取 top_n（不做唯一解仲裁）
        # 返回 [(rule_dict, 命中条件数)]，调用方取 meaning / risk_warning 组装研报表述
        hits = []
        for rule in self.rules:
            conds = rule["conditions"]
            if conds and all(self._check(c, snapshot) for c in conds):
                hits.append((rule, len(conds)))
        hits.sort(key=lambda t: -t[1])
        if hits:
            logger.info("组合规则命中 %s 条: %s", len(hits),
                        [(r["id"], n) for r, n in hits[:top_n]])
        return hits[:top_n]

    def risk_flags(self, series: dict, industry: str) -> list:
        # 风控第1层硬规则：同比相对变化绝对值超行业阈值 → 异动清单（零 LLM）
        # 返回 [{indicator, change, rel_change, limit}] 按变动幅度降序，空列表 = 无异动
        limits = self._industry_limits(industry)
        flags = []
        for ind, vals in series.items():
            if len(vals) < 2:
                continue
            cur, prev = float(vals[-1]), float(vals[-2])
            if prev == 0:
                continue
            rel = abs((cur - prev) / prev)
            limit = float(limits.get(ind, limits.get("default", 0.5)))
            if rel > limit:
                flags.append({"indicator": ind, "change": round(cur - prev, 4),
                              "rel_change": round(rel, 4), "limit": limit})
        flags.sort(key=lambda f: -f["rel_change"])
        return flags

    def _industry_limits(self, industry: str) -> dict:
        # 行业阈值合并：行业段覆盖同名项，未覆盖项回落全局 default
        merged = {"default": self._risk.get("default", 0.5)}
        merged.update((self._risk.get("industries") or {}).get(industry) or {})
        return merged

    def _check(self, cond: dict, snapshot: dict) -> bool:
        # 单条件校验；两类条件：
        #   绝对值条件 {indicator, op: gt/lt, value, [persist: N]}——对最近 N 期取值逐一校验
        #   方向条件   {indicator, direction: up/down/stable, [magnitude: large], [persist: N]}
        snap = snapshot.get(cond["indicator"])
        if snap is None:
            return False
        if "op" in cond:
            need = int(cond.get("persist", 1))
            window = snap["values"][-need:]
            val = float(cond["value"])
            if cond["op"] == "gt":
                return all(v > val for v in window)
            if cond["op"] == "lt":
                return all(v < val for v in window)
            return False
        if snap["direction"] != cond["direction"]:
            return False
        if cond.get("magnitude") and snap["magnitude"] != cond["magnitude"]:
            return False
        # persist 默认 0：方向匹配即足够；stable 的 persist 天然为 0，默认 1 会恒假
        return snap["persist"] >= int(cond.get("persist", 0))

    def _persist_count(self, vals: list, ind: str) -> int:
        # 从最新期向前数连续同向期数；stable 视为中断；同向切换立即截断
        ind_type = self.indicator_types.get(ind, "rel")
        count, last = 0, 0
        for i in range(len(vals) - 1, 0, -1):
            moved = self._step_move(vals[i], vals[i - 1], ind_type)
            if moved == 0:
                break
            if last == 0:
                last, count = moved, 1
            elif moved == last:
                count += 1
            else:
                break
        return count

    def _step_move(self, cur: float, prev: float, ind_type: str) -> int:
        # 相邻两期动向：+1 上行 / -1 下行 / 0 稳定
        if ind_type == "pp":
            d = cur - prev
            return 0 if abs(d) <= self.stable_pp else (1 if d > 0 else -1)
        rel = self._rel(cur, prev)
        if rel == float("inf"):
            return 1 if cur > prev else -1
        if abs(rel) <= self.stable_rel:
            return 0
        return 1 if rel > 0 else -1

    @staticmethod
    def _rel(cur: float, prev: float) -> float:
        # 相对变化率；prev=0 无意义返回 inf，由 _step_move 按符号兜底
        return (cur - prev) / abs(prev) if prev != 0 else float("inf")
