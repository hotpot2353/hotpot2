# Rerank 客户端：VikingDB 知识库数据面 /api/knowledge/service/rerank（doubao-seed-rerank）
# 鉴权：IAM AK/SK HMAC 签名（.env VIKINGDB_AK/VIKINGDB_SK），该接口明确不支持方舟 API Key
# 职责边界：只管"调 API 拿得分"（重试/得分契约校验/token 账目），候选集构建与重排序由调用方负责
# SDK 延迟导入：vikingdb 仅在实例化/调用时才需要——纯召回层评测（rerank=false）无 SDK 也可运行
import random
import time
from functools import lru_cache

from core.config import get_env_var, get_settings
from core.logger import get_logger

logger = get_logger(__name__)


class RerankClient:
    # rerank 客户端：SDK IAM 鉴权 + 指数退避重试；返回得分与输入 texts 顺序一一对应
    def __init__(self, retries: int = 3):
        # retries: 最大尝试次数；可重试面 = 网络异常 + 429 配额(300004) + 5xx 过载(1000028)
        from vikingdb.auth import IAM
        from vikingdb.knowledge import VikingKnowledge

        vik = get_settings().vikingdb
        ak = get_env_var("VIKINGDB_AK")
        sk = get_env_var("VIKINGDB_SK")
        if not ak or not sk:
            raise RuntimeError("VIKINGDB_AK/VIKINGDB_SK 未配置：请检查项目根 .env")
        self.retries = retries
        self.client = VikingKnowledge(
            host=vik.host, region=vik.region,
            auth=IAM(ak=ak, sk=sk), scheme=vik.scheme, timeout=vik.timeout)
        # 累计 token 消耗（实例生命周期），成本评测脚本读取
        self.total_tokens = 0

    def scores(self, query: str, texts: list, model: str,
               instruction: str = "") -> list:
        # 批量计算 texts 各自与 query 的相关性得分，返回与输入顺序一一对应的 float 列表
        # model: 显式 rerank 模型名（rag.yaml retrieval.rerank_model）
        # instruction: 重排指令，仅 doubao-seed-rerank 生效；空串不传走服务端默认
        from vikingdb.exceptions import VikingException
        from vikingdb.knowledge.models.rerank import RerankDataItem

        if not texts:
            return []
        datas = [RerankDataItem(query=query, content=t) for t in texts]
        resp = None
        for attempt in range(1, self.retries + 1):
            try:
                resp = self.client.rerank(
                    datas=datas, rerank_model=model,
                    rerank_instruction=instruction or None)
                break
            except VikingException as exc:
                # 429 配额限流与 5xx 服务过载可重试；其余 4xx（未下单 300001/鉴权失败/
                # 参数错误）确定性失败，重试无意义立即上抛
                retryable = exc.status_code == 429 or (exc.status_code or 0) >= 500
                if not retryable or attempt == self.retries:
                    raise
                logger.warning("rerank 请求失败(%s/%s): %s", attempt, self.retries, exc)
                time.sleep(2 ** attempt + random.uniform(0, 0.5))
            except Exception as exc:
                # 网络层异常（超时/连接失败）：可重试
                if attempt == self.retries:
                    raise
                logger.warning("rerank 网络异常(%s/%s): %s", attempt, self.retries, exc)
                time.sleep(2 ** attempt + random.uniform(0, 0.5))
        # 响应契约校验在重试循环外：业务错误/契约破裂属确定性失败，重试无意义
        if resp.code not in (0, None):
            raise RuntimeError(f"rerank 业务错误 code={resp.code}: {resp.message}")
        result = resp.result
        if result is None or len(result.scores) != len(texts):
            got = len(result.scores) if result else 0
            raise ValueError(f"rerank 得分数契约破裂: 期望 {len(texts)} 实得 {got}")
        self.total_tokens += int(result.token_usage or 0)
        return list(result.scores)


@lru_cache(maxsize=4)
def get_rerank_client(retries: int = 3) -> RerankClient:
    # 按参数缓存单例：同参数重复获取返回同一实例
    # 业务参数（model/instruction）由调用方从 rag.yaml 取，本模块不反向依赖 rag
    return RerankClient(retries=retries)
