# FastAPI 服务层：把 CLI 入口升级为 HTTP 接口
# 零侵入：七节点 DAG 与各 Agent 完全不感知 HTTP 的存在；本文件只做「接单 → 跑 Flow → 回 JSON」
# 同步注意：run_flow 是 ~25s 同步长任务，接口必须用普通 def（FastAPI 自动丢线程池执行），
#   写成 async def 里直接调会卡死事件循环，整个服务无响应
# 串行锁：报告单写者写 reports/{code}_report.md 固定文件名，并发研判同标的会互踩——
#   全局锁串行化
# 报告查看：/ 静态前端 + /api/reports 列表与正文——纯文件读取零 LLM 零 DB，
#   与研判链路互不感知；Flow 懒导入：报告查看不依赖 crewai，编排层缺依赖时服务仍可起
import re
import sys
import threading
from pathlib import Path

# src/ 入 sys.path：uvicorn src.api:app 方式启动时，core/agents/llm 等按 src 根绝对导入
sys.path.insert(0, str(Path(__file__).resolve().parent))

import markdown  # noqa: E402
from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from core.config import get_settings  # noqa: E402
from core.logger import get_logger  # noqa: E402
from core.paths import REPORTS_DIR  # noqa: E402

# 入口先触发 load_dotenv（挂在 core.config 导入时），与 main.py 同款防 crewai trace 询问
_ = get_settings

logger = get_logger(__name__)

app = FastAPI(
    title="fin_analyst_crew",
    description="多维研判服务：POST /analyze 同步执行全链路（约 25s），/ 为报告查看前端，/docs 有自助调试页",
)
# 报告单写者互斥：同标的研究报告文件名固定，全局锁串行
_lock = threading.Lock()


@app.exception_handler(ValueError)
async def on_value_error(request: Request, exc: ValueError) -> JSONResponse:
    # 链路主动拒绝（如意图识别未解析出股票代码）→ 400 + 可读原因，前端直接展示
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def on_uncaught(request: Request, exc: Exception) -> JSONResponse:
    # 兜底：未捕获异常统一回 JSON，避免纯文本 500 让前端 JSON 解析二次报错
    logger.exception("[api] 未捕获异常 %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": f"服务内部错误：{exc}"})

# 前端静态页（纯原生实现零依赖）；报告文件名白名单：防路径穿越，只放行库内产物
_WEB_INDEX = Path(__file__).resolve().parent / "web" / "index.html"
_SAFE_REPORT = re.compile(r"^[A-Za-z0-9_\-]+\.md$")


class QueryReq(BaseModel):
    query: str


def _run(query: str) -> dict:
    # 与 flow.run_flow 同款构造；不复用 run_flow 是为了拿到完整 state（trace_id/report_path）
    # Flow 懒导入：编排层（crewai）仅研判时需要，缺失时给可读错误而非服务起不来
    try:
        from agents.orchestrator.flow import ResearchFlow
        from agents.orchestrator.state import AgentState
    except ModuleNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"研判编排层依赖缺失（{exc.name}），报告查看不受影响；"
                   f"请按 pyproject 安装依赖后重试") from exc
    with _lock:
        logger.info("[api] 收到研判请求 query=%r", query)
        flow = ResearchFlow(initial_state=AgentState(query=query))
        summary = flow.kickoff()
    state = flow.state
    return {
        "summary": summary,
        "trace_id": state.trace.get("trace_id", ""),
        "report_path": state.report_path or "",
        "errors": len(state.errors),
    }


def _report_meta(path: Path) -> dict:
    # 报告元信息：标题取首个一级标题，生成时间取 meta 行；缺省回落文件名
    text = path.read_text(encoding="utf-8")
    title = path.stem
    generated = ""
    for line in text.splitlines():
        if line.startswith("# ") and title == path.stem:
            title = line[2:].strip()
        if line.startswith(">") and "生成时间" in line and not generated:
            hit = re.search(r"生成时间\s+([0-9\-: ]+)", line)
            generated = hit.group(1).strip() if hit else ""
    return {"file": path.name, "title": title, "generated_at": generated,
            "is_stub": "stub" in path.name, "size": path.stat().st_size}


@app.get("/")
def index() -> FileResponse:
    # 报告查看前端（原生 HTML/JS，零外部依赖）
    return FileResponse(_WEB_INDEX, media_type="text/html")


@app.get("/api/reports")
def list_reports() -> dict:
    # 报告列表：正报告在前按修改时间倒序，stub 演示稿垫底置灰
    items = [_report_meta(p) for p in REPORTS_DIR.glob("*.md")]
    items.sort(key=lambda x: (x["is_stub"], x["generated_at"] or ""), reverse=True)
    items.sort(key=lambda x: x["is_stub"])  # 稳定排序：stub 恒垫底
    return {"reports": items}


@app.get("/api/reports/{name}")
def get_report(name: str) -> dict:
    # 报告正文：原文 + 服务端渲染 HTML（tables 扩展支撑风险清单表格）
    if not _SAFE_REPORT.fullmatch(name):
        raise HTTPException(status_code=400, detail="非法文件名")
    path = REPORTS_DIR / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"报告不存在: {name}")
    raw = path.read_text(encoding="utf-8")
    html = markdown.markdown(raw, extensions=["tables"])
    return {**_report_meta(path), "raw": raw, "html": html}


@app.post("/analyze")
def analyze(req: QueryReq) -> dict:
    # 生产入口：POST {"query": "..."}；普通 def 保证长任务进线程池
    return _run(req.query)


@app.get("/analyze")
def analyze_get(query: str) -> dict:
    # 浏览器验收便利入口（地址栏直接带中文 query 自动转码）；生产环境请用 POST
    return _run(query)


@app.get("/health")
def health() -> dict:
    # 存活探针：不跑链路，只确认服务在
    return {"status": "ok"}


if __name__ == "__main__":
    # 一键启动：python src\api.py（与 streamlit run 同款体验，等价于 uvicorn src.api:app）
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8317)
