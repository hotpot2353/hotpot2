# 股票池：池清单来自 data 层既有行情快照（50 只），不新建清单文件
# 池外标的走显式降级设计：缺维度必须声明，用户必须知情
import csv
from difflib import get_close_matches

from core.paths import DATA_ROOT

_SNAPSHOT_CSV = DATA_ROOT / "raw" / "structured" / "quotes" / "snapshot.csv"

# 进程内缓存：[(code, name), ...]，首次访问读一次 csv
_POOL = None


def _load() -> list:
    # 快照 csv 带 BOM 头，用 utf-8-sig 读；文件缺失视为空池（上层按池外降级）
    global _POOL
    if _POOL is None:
        rows = []
        if _SNAPSHOT_CSV.exists():
            with open(_SNAPSHOT_CSV, "r", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    rows.append((row["code"].strip(), row["name"].strip()))
        _POOL = rows
    return _POOL


def contains(code: str) -> bool:
    # 是否池内标的
    return any(c == code for c, _ in _load())


def suggest(name: str) -> str:
    # 池内近似标的提示：先子串匹配（"平安银行"→"中国平安"），再模糊匹配
    if not name:
        return ""
    pool_names = [n for _, n in _load()]
    for n in pool_names:
        if name in n or n in name:
            return n
    close = get_close_matches(name, pool_names, n=1, cutoff=0.4)
    return close[0] if close else ""
