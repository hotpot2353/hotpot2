# 研判主 Flow：CrewAI Flows 确定性骨架——DAG 硬编码，LLM 只做节点内推理，不做任务拆分
# 链路：意图识别 → 数据采集 → 三分析并行 → and_ 汇合 → 风控 → 综合建议 → 报告 → 收尾（八节点）
# 依赖顺序：采集完成才启动分析层；分析层全部完成才进风控；风控完成才出综合建议；
#   建议消费全部结构化结论（评级/时机/纪律为代码推导），报告在建议之后渲染
import json
import time

from crewai.flow.flow import Flow, and_, listen, start

from agents.advice.agent import run as _advice
from agents.collector.agent import run as _collect
from agents.fundamental.agent import run as _fundamental
from agents.orchestrator.intent import run as _intent
from agents.orchestrator.state import AgentState
from agents.report.agent import run as _report
from agents.risk.agent import run as _risk
from agents.sentiment.agent import run as _sentiment
from agents.technical.agent import run as _technical
from core.logger import get_logger
from core.paths import LOGS_DIR

logger = get_logger(__name__)


def _timed(node: str, state: AgentState, fn, *args, **kwargs):
    # 节点计时包装：耗时写 trace["nodes"][node]["ms"]
    # tokens 等细项由 Agent 内部回填（chat usage 参数），本函数只管计时——finally 保证异常也留痕
    t0 = time.perf_counter()
    try:
        return fn(*args, **kwargs)
    finally:
        ms = round((time.perf_counter() - t0) * 1000)
        entry = state.trace["nodes"].setdefault(node, {})
        entry["ms"] = ms
        logger.info("[trace] %s %s 耗时 %dms", state.trace["trace_id"], node, ms)


class ResearchFlow(Flow[AgentState]):
    # 八节点确定性 DAG；节点方法只做「调 Agent → 写 state」，业务全在 Agent 侧

    @start()
    def intent_recognize(self) -> None:
        # 步骤1 意图识别：LLM 单点推理 query → 六字段 JSON（orchestrator_intent 提示词）
        logger.info("[1/8] 意图识别 query=%r", self.state.query)
        self.state.intent = _timed("intent", self.state, _intent, self.state.query, self.state)
        logger.info("[1/8] 意图识别完成 %s", self.state.intent)

    @listen(intent_recognize)
    def collect(self) -> None:
        # 步骤2 数据采集：AKShare/a-stock-data 工具调用 + 清洗 + 降级
        logger.info("[2/8] 数据采集")
        self.state.raw_data = _timed("collect", self.state, _collect, self.state)

    @listen(collect)
    def analyze_fundamental(self) -> None:
        # 3.1 基本面：规则引擎组合研判（零 LLM）+ LLM 叙述要点提炼
        logger.info("[3/8] 基本面分析")
        self.state.fundamental = _timed("fundamental", self.state, _fundamental, self.state)

    @listen(collect)
    def analyze_technical(self) -> None:
        # 3.2 技术面：行情时序纯代码解析 + LLM 两段叙述
        logger.info("[3/8] 技术面分析")
        self.state.technical = _timed("technical", self.state, _technical, self.state)

    @listen(collect)
    def analyze_sentiment(self) -> None:
        # 3.3 舆情：LLM 标题三分类打标 + 代码聚合
        logger.info("[3/8] 舆情分析")
        self.state.sentiment = _timed("sentiment", self.state, _sentiment, self.state)

    @listen(and_(analyze_fundamental, analyze_technical, analyze_sentiment))
    def risk_check(self) -> None:
        # 步骤4 风控：and_ 汇合保证三分析全部完成后才执行
        logger.info("[4/8] 风控校验")
        self.state.risk = _timed("risk", self.state, _risk, self.state)

    @listen(risk_check)
    def investment_advice(self) -> None:
        # 步骤5 综合建议：跨维度确定性评分定级（评级/时机/纪律代码产出）+ LLM 仅叙述核心逻辑；
        # 消费风控之后的全量结构化结论，报告渲染其上游
        logger.info("[5/8] 综合研判建议")
        self.state.advice = _timed("advice", self.state, _advice, self.state)

    @listen(investment_advice)
    def generate_report(self) -> None:
        # 步骤6 报告：唯一写 reports/ 的节点（单写者）；纯模板渲染零 LLM
        logger.info("[6/8] 研报生成")
        out = _timed("report", self.state, _report, self.state)
        self.state.report_path = out["report_path"]
        self.state.summary = out["summary"]

    @listen(generate_report)
    def finalize(self) -> str:
        # 步骤7 收尾：精简结论返回终端用户；追加全链路 trace 汇总
        nodes = self.state.trace.get("nodes", {})
        total_ms = sum(n.get("ms", 0) for n in nodes.values())
        prompt = sum((n.get("tokens") or {}).get("prompt", 0) for n in nodes.values())
        completion = sum((n.get("tokens") or {}).get("completion", 0) for n in nodes.values())
        reasoning = sum((n.get("tokens") or {}).get("reasoning", 0) for n in nodes.values())
        # 证据引用计数：结构化链路恒为 0，RAG 链路=各维度命中年报子块数
        refs = sum(len((getattr(self.state, k) or {}).get("evidence_ref") or [])
                   for k in ("fundamental", "technical", "sentiment"))
        # trace 落盘：全量账单存 logs/trace/{trace_id}.json，
        # run_ab/check_fidelity 聚合消费；落盘失败只告警不影响链路
        try:
            trace_dir = LOGS_DIR / "trace"
            trace_dir.mkdir(parents=True, exist_ok=True)
            # errors 在 state.errors 不在 trace 里，落盘时合并（评测账单需要 errors 列）
            (trace_dir / f"{self.state.trace['trace_id']}.json").write_text(
                json.dumps({**self.state.trace, "errors": self.state.errors},
                           ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.warning("[trace] 落盘失败(不影响链路): %s", exc)
        logger.info("[trace] %s 汇总 节点=%s 总耗时=%dms prompt=%d completion=%d reasoning=%d "
                    "evidence_refs=%d errors=%d",
                    self.state.trace["trace_id"], sorted(nodes), total_ms,
                    prompt, completion, reasoning, refs, len(self.state.errors))
        logger.info("[7/8] 收尾 report=%s", self.state.report_path)
        logger.info("[8/8] 流程完成")
        return self.state.summary


def run_flow(query: str) -> str:
    # 对外入口：构造状态 → 跑完整 DAG → 返回精简结论
    # crewai 1.15.x 运行时 Flow 继承 pydantic BaseModel，构造只收关键字参数，
    # state 走 initial_state 字段（内部 _create_initial_state 校验后挂到 self.state）
    flow = ResearchFlow(initial_state=AgentState(query=query))
    return flow.kickoff()
