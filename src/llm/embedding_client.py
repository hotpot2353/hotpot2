# 嵌入客户端：文本 → 向量（火山方舟 multimodal embeddings 接口）
# 职责边界：只管"调 API 拿向量"（限流/重试/维度契约校验），切块与入库由调用方负责
# 实测口径（模型 doubao-embedding-vision-251215）：
#   1. vision 系列仅支持 /embeddings/multimodal；标准 /embeddings 报 InvalidParameter 不支持
#   2. multimodal 的 input 列表为联合嵌入语义：整列表 → 单个混合向量（服务跨模态匹配），
#      因此批量文本嵌入必须逐条请求，无批接口
#   3. 同文本多次请求 cosine≈0.9977（服务端轻微非确定性）——本项目去重在嵌入前按哈希完成，
#      不依赖向量逐位一致，无影响
# 版本锚定：model 必须传显式 Model ID——别名 doubao-embedding-vision 直调报 NotFound，
#   且存量向量与生成模型版本绑定，混存不可比
import random
import time
from functools import lru_cache

import requests

from core.config import get_settings
from core.logger import get_logger

logger = get_logger(__name__)

# multimodal 端点相对路径；base_url 来自 config.yaml ark 段（OpenAI 兼容 /api/v3）
_EMBEDDING_PATH = "/embeddings/multimodal"

# 可重试状态码：限流与网关/服务端瞬时故障；其余 4xx 属客户端错误，重试无意义
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# 逐条模式下的进度日志间隔
_LOG_EVERY = 100


class EmbeddingClient:
    # 嵌入客户端：限流间隔 + 指数退避重试 + 维度契约校验
    def __init__(self, model: str, dimension: int,
                 interval: float = 0.1, retries: int = 3):
        # model: 显式 Model ID，防别名漂移
        # dimension: 输出维度，须与 chunks 表 vector(N) 列定义一致
        # interval: 两次请求最小间隔（秒）；模型 RPM 上限 15k，0.1s≈600 RPM 远低于限流，
        #   仅作礼貌限流给共享额度的其他调用留余量
        # retries: 最大尝试次数
        ark = get_settings().ark
        if not ark.api_key:
            raise RuntimeError("ARK_API_KEY 未配置：请检查项目根 .env")
        self.model = model
        self.dimension = dimension
        self.interval = interval
        self.retries = retries
        self.timeout = ark.timeout
        self.url = ark.base_url + _EMBEDDING_PATH
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {ark.api_key}"})
        self._last = 0.0
        # 累计 token 消耗（实例生命周期），成本评测脚本读取
        self.total_prompt_tokens = 0

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
                logger.warning("嵌入请求网络异常(%s/%s): %s", attempt, self.retries, exc)
                time.sleep(2 ** attempt)
                continue
            if resp.status_code < 400:
                return resp.json()
            detail = f"HTTP {resp.status_code}: {resp.text[:200]}"
            if resp.status_code not in _RETRYABLE_STATUS or attempt == self.retries:
                # 4xx 客户端错误（如 ModelNotOpen/鉴权失败）或重试额度耗尽：直接抛
                raise requests.HTTPError(detail, response=resp)
            logger.warning("嵌入请求失败(%s/%s): %s", attempt, self.retries, detail)
            time.sleep(2 ** attempt)
        raise RuntimeError("unreachable")

    def embed_texts(self, texts: list) -> list:
        # 批量入口：逐条请求（该模型无批接口，见模块头注释），返回与输入顺序对应的向量列表
        # 失败即抛（含第几条的上下文），由调用方按 manifest 断点重跑，不做半途截断
        if not texts:
            return []
        out = []
        for i, text in enumerate(texts, start=1):
            payload = {
                "model": self.model,
                "input": [{"type": "text", "text": text}],
                "dimensions": self.dimension,
            }
            data = self._post(payload)
            # multimodal 返回 data 为 dict（联合嵌入单向量形态）：{embedding: [...], object: ...}
            vec = data.get("data", {}).get("embedding")
            if not vec or len(vec) != self.dimension:
                got = len(vec) if vec else 0
                raise ValueError(
                    f"向量维度契约破裂(第 {i}/{len(texts)} 条): 期望 {self.dimension} 实得 {got}，中止以防脏数据")
            out.append(vec)
            tokens = int(data.get("usage", {}).get("prompt_tokens", 0))
            self.total_prompt_tokens += tokens
            if i % _LOG_EVERY == 0 or i == len(texts):
                logger.info("嵌入进度: %s/%s, 本轮累计 tokens=%s",
                            i, len(texts), self.total_prompt_tokens)
        return out


@lru_cache(maxsize=4)
def get_embedding_client(model: str, dimension: int,
                         interval: float = 0.1,
                         retries: int = 3) -> EmbeddingClient:
    # 按参数缓存单例：同参数重复获取返回同一实例
    # 业务参数（model/dimension）由调用方从 rag.yaml 取，本模块不反向依赖 rag
    return EmbeddingClient(model=model, dimension=dimension,
                           interval=interval, retries=retries)
