# Flow 全局状态：全链路唯一共享数据结构，Agent 间通信统一结构化 JSON
# 各节点只读上游字段、写自己的字段；errors 记录异常与缺失，失败不中断
from uuid import uuid4

from pydantic import BaseModel, Field


def _new_trace() -> dict:
    # 每次构造 state 生成全新 trace 骨架：trace_id 串联一次研判的全部日志，nodes 按节点记账
    return {"trace_id": uuid4().hex[:8], "nodes": {}}


class AgentState(BaseModel):
    # query: 用户原始请求
    query: str
    # intent: 步骤1 意图识别输出（stock_code/stock_name/years/dimensions/output/extra）
    intent: dict = Field(default_factory=dict)
    # raw_data: 步骤2 数据采集输出（financials/quotes/news/missing），按域切片下发
    raw_data: dict = Field(default_factory=dict)
    # fundamental: 3.1 基本面分析 JSON
    fundamental: dict = Field(default_factory=dict)
    # technical: 3.2 技术面分析 JSON
    technical: dict = Field(default_factory=dict)
    # sentiment: 3.3 舆情分析 JSON
    sentiment: dict = Field(default_factory=dict)
    # risk: 步骤4 风控三层校验输出
    risk: dict = Field(default_factory=dict)
    # advice: 步骤5 综合建议（评级/评分/时机/信号/纪律为代码推导，核心逻辑为 LLM 叙述）
    advice: dict = Field(default_factory=dict)
    # report_path: 步骤6 研报落盘路径（单写者：仅 report 节点写 reports/）
    report_path: str = ""
    # summary: 步骤7 精简结论，返回终端用户
    summary: str = ""
    # errors: 全链路异常/数据缺失标记，供风控完整性校验与研报局限性声明
    errors: list = Field(default_factory=list)
    # trace: 可观测段：{trace_id, nodes: {节点: {ms, tokens, ...}}}
    # 约定：flow 侧写 ms（计时），Agent 侧写 tokens（usage 回填），两边 setdefault 互不覆盖
    trace: dict = Field(default_factory=_new_trace)
