# 聊天客户端：对话推理（火山方舟 OpenAI 兼容 /chat/completions 接口）
# 职责边界：只管"发消息拿回复"（限流/重试/token 记账/思考开关），提示词组装与 JSON 解析由调用方负责
# 实测口径（模型 doubao-seed-2-1-pro-260628）：
#   1. 别名 doubao-seed-2.1-pro 直调 404 InvalidEndpointOrModel.NotFound——必须显式 Model ID
#      （config.yaml 已锁显式 ID）
#   2. 属思考模型：极简请求也产生 reasoning_tokens（实测 trivial prompt 约占总 token 1/3），
#      轻任务（意图识别等）可 thinking="disabled" 关闭思考；reasoning_tokens 单独记账供成本评测
import random
import time
from functools import lru_cache

import requests

from core.config import get_settings
from core.logger import get_logger

logger = get_logger(__name__)

# chat 端点相对路径；base_url 来自 config.yaml ark 段（OpenAI 兼容 /api/v3）
_CHAT_PATH = "/chat/completions"

# 可重试状态码：限流与网关/服务端瞬时故障；其余 4xx 属客户端错误，重试无意义
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class ChatClient:
    # 对话客户端：礼貌限流 + 指数退避重试 + token 分项记账
    def __init__(self, model: str, temperature: float = 0.3,
                 interval: float = 0.0, retries: int = 3):
        # model: 显式 Model ID，防别名漂移
        # temperature: 生成随机度；研判类任务低温保稳定，调用方可按次覆盖
        # interval: 两次请求最小间隔（秒）；查询期对话低频单发，默认 0 不限流
        # retries: 最大尝试次数
        ark = get_settings().ark
        if not ark.api_key:
            raise RuntimeError("ARK_API_KEY 未配置：请检查项目根 .env")
        self.model = model
        self.temperature = temperature
        self.interval = interval
        self.retries = retries
        self.timeout = ark.timeout
        self.url = ark.base_url + _CHAT_PATH
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {ark.api_key}"})
        self._last = 0.0
        # 累计 token 消耗（实例生命周期），成本评测脚本读取；reasoning 单列便于评估思考开销
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_reasoning_tokens = 0

    def _throttle(self) -> None:
        # 保证两次请求间至少间隔 interval 秒，附随机抖动避免节奏化访问
        wait = self.interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait + random.uniform(0, 0.05))
        self._last = time.time()

    def _post(self, payload: dict) -> dict:
        # 单次请求 + 重试：网络异常与 429/5xx 指数退避重试；其余 4xx 客户端错误直接上抛（重试无意义）
        for attempt in range(1, self.retries + 1):
            self._throttle()
            try:
                resp = self.session.post(self.url, json=payload, timeout=self.timeout)
            except requests.RequestException as exc:
                # 网络层异常（超时/连接失败）：可重试
                if attempt == self.retries:
                    raise
                logger.warning("chat 请求网络异常(%s/%s): %s", attempt, self.retries, exc)
                time.sleep(2 ** attempt)
                continue
            if resp.status_code < 400:
                return resp.json()
            detail = f"HTTP {resp.status_code}: {resp.text[:200]}"
            if resp.status_code not in _RETRYABLE_STATUS or attempt == self.retries:
                # 4xx 客户端错误（如 NotFound/鉴权失败）或重试额度耗尽：直接抛
                raise requests.HTTPError(detail, response=resp)
            logger.warning("chat 请求失败(%s/%s): %s", attempt, self.retries, detail)
            time.sleep(2 ** attempt)
        raise RuntimeError("unreachable")

    def chat(self, messages: list, temperature: float = None,
             max_tokens: int = None, thinking: str = None, usage: dict = None) -> str:
        # 单轮对话入口：messages=[{"role": "system"/"user"/"assistant", "content": str}]
        # 返回助手回复文本；失败即抛，token 记账只对成功请求累计
        # temperature: None 用实例默认；max_tokens: None 不传走服务端默认
        # thinking: None 服务端默认 / "enabled" / "disabled"——思考模型按任务精度需求开关
        # usage: 传入 dict 则回填本次请求的分项 token（trace 节点记账用）；
        #   走参数回填而非读实例累计——并行节点共享 lru_cache 单例，实例累计拆不到节点
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if thinking is not None:
            payload["thinking"] = {"type": thinking}
        data = self._post(payload)
        choices = data.get("choices") or []
        if not choices:
            raise ValueError(f"chat 响应契约破裂：choices 为空，resp={str(data)[:200]}")
        usage_data = data.get("usage", {}) or {}
        prompt_tokens = int(usage_data.get("prompt_tokens", 0))
        completion_tokens = int(usage_data.get("completion_tokens", 0))
        details = usage_data.get("completion_tokens_details") or {}
        reasoning_tokens = int(details.get("reasoning_tokens", 0) or 0)
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_reasoning_tokens += reasoning_tokens
        if usage is not None:
            usage["prompt"] = prompt_tokens
            usage["completion"] = completion_tokens
            usage["reasoning"] = reasoning_tokens
        content = choices[0].get("message", {}).get("content", "")
        logger.info("chat 完成 model=%s prompt=%s completion=%s reasoning=%s",
                    self.model, prompt_tokens, completion_tokens, reasoning_tokens)
        return content


@lru_cache(maxsize=8)
def get_chat_client(model: str, temperature: float = 0.3,
                    interval: float = 0.0, retries: int = 3) -> ChatClient:
    # 按参数缓存单例：同参数重复获取返回同一实例
    # model 由调用方从 get_settings().ark.llm_model 取（显式 ID），本模块不重复读配置
    return ChatClient(model=model, temperature=temperature,
                      interval=interval, retries=retries)
