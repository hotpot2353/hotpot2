# 年报规范化规则：集中管理常量、预编译正则与判定工具
# - 规则全部基于对 600519(白酒)/601318(保险)/600030(券商) 三份样本的实测
# - 阈值可被 configs/config.yaml 的 normalize 段覆盖
import re

NORMALIZER_VERSION = "normalize/6"

# 中文数字序号 -> 阿拉伯数字（章节/条目压栈用）
CN_DIGITS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6,
             "七": 7, "八": 8, "九": 9, "十": 10}

# parser 输出的物理页边界
PAGE_MARK_RE = re.compile(r"^##\s*第\s*(\d+)\s*页\s*$")

# Markdown hash 标题（parser 已识别的节/小节/子条）
HASH_SECTION_RE = re.compile(r"^##\s*第[一二三四五六七八九十百0-9]+节[\s　]+(.{1,40}?)\s*$")
HASH_L3_RE = re.compile(r"^###\s*([一二三四五六七八九十]+)、\s*(.{1,40}?)\s*$")
HASH_L4_RE = re.compile(r"^####\s*（[一二三四五六七八九十]+）")

# 两行式节标题的首行：单独成行的「第X节」
SECTION_KEY_RE = re.compile(r"^(第[一二三四五六七八九十百0-9]+节)\s*$")
# 已带标题的单行节标题（兼容全角/半角空格）
SECTION_INLINE_RE = re.compile(r"^(第[一二三四五六七八九十百0-9]+节)[\s　]+(.{1,40}?)\s*$")
# 正文里独占一行的虚拟 L4 子条（表区间外才生效）
VIRTUAL_L4_RE = re.compile(r"^（[一二三四五六七八九十]+）[^，。；]{1,40}$")
# L3 序号前缀
L3_PREFIX_RE = re.compile(r"^([一二三四五六七八九十]+)、\s*(.{1,40}?)\s*$")

# 页眉页脚（家具）
FOOTER_SLASH_RE = re.compile(r"^\d+\s*/\s*\d+$")               # 茅台 58 / 143
FOOTER_DASH_RE = re.compile(r"^-\s*\d+\s*-$")                  # 美的 - 1 -
INNER_PAGE_RE = re.compile(r"^第\s*\d+\s*页(?:[，,]\s*共\s*\d+\s*页)?$")  # 中信 第 1 页 / 第 9 页，共 9 页
PINGAN_BOOK_RE = re.compile(r"^二零二[三四]年年报$")           # 平安书眉
COMPANY_TAIL_RE = re.compile(r"^[\u4e00-\u9fa5（）()]{2,30}(?:公司|集团)\s*\d{0,4}$")  # 公司名，可粘连印刷页码
ANNUAL_HEADER_RE = re.compile(r"^.*公司\s*20\d{2}\s*年年度报告$")  # 茅台 XX公司2024 年年度报告
# 紧邻家具时才删除的孤立印刷页码（避免误删表内附注号/数值）
BARE_PAGE_NO_RE = re.compile(r"^\d{1,3}$")

# 目录
TOC_TITLE_RE = re.compile(r"^目录$")
# 茅台型点引导线条目：「释义 ...... 4」或「第一节 标题 ...... - 3 -」或「释义 ......」（无页码）
# 至少 30 个点/句号才判定（排除正文省略号如「肿瘤 ........」）
TOC_LEADER_RE = re.compile(r"[\.·⋯…]{30,}\s*(?:-\s*\d{1,3}\s*-|\d{1,3})?\s*$")
# 目录条目配对的孤立页码：阿拉伯 1-3 位或罗马数字
TOC_PAGE_RE = re.compile(r"^(?:\d{1,3}|[ivx]{1,6})$")
# 目录页码范围：1-6 / 7-8 / 25-180 / A-1（平安银行等审计报告目录格式）
TOC_PAGE_RANGE_RE = re.compile(r"^(?:\d{1,3}\s*[-–—]\s*\d{1,3}|[A-Z]\s*[-–—]\s*\d{1,3})$")
# 目录短标题长度上限
TOC_TITLE_MAX_LEN = 24

# 主表标题（scope: merge/parent，kind: bs/is/cf/eq）
_STATEMENT_KINDS = {
    "资产负债表": "bs",
    "利润表": "is",
    "现金流量表": "cf",
    "所有者权益变动表": "eq",
    "股东权益变动表": "eq",
}
CONTINUATION_RE = re.compile(r"[（(]\s*续\s*[）)]\s*$")
STATEMENT_LINE_RE = re.compile(
    r"^(合并|母公司|公司)?\s*(合并)?\s*"
    r"(资产负债表|利润表|现金流量表|所有者权益变动表|股东权益变动表)"
    r"\s*(?:[（(]\s*续\s*[）)])?\s*$"
)

# 表尾标记（弱控制：命中仅打 kind=tail，不立即弹表）
TAIL_MARKERS = (
    "资产总计", "负债合计", "负债和所有者权益", "负债和股东权益", "负债及股东权益",
    "所有者权益（或股东权益）合计", "股东权益合计", "所有者权益合计",
    "经营活动产生的现金流量净额", "现金及现金等价物净增加额", "现金及现金等价物净减少额",
    "现金及现金等价物余额", "期末现金及现金等价物余额",
    "综合收益总额", "本期期末余额",
)

# 金融文本流：数字 token（千分位/小数/括号负数）与空值破折号
NUMBER_TOKEN_RE = re.compile(r"^\(?-?[\d,]+(?:\.\d+)?\)?$")
DASH_NULL_RE = re.compile(r"^[–\-—－]{1,2}$")
# 附注号：1-3 位整数
NOTE_TOKEN_RE = re.compile(r"^\d{1,3}$")
# 科目名：中文/字母开头，长度受限，不含日期单位词
SUBJECT_LINE_RE = re.compile(r"^[\u4e00-\u9fa5A-Za-z][\u4e00-\u9fa5A-Za-z0-9（）()、：:，,\.\s]{0,38}[\u4e00-\u9fa5A-Za-z0-9）)]$")
# 无值分组标签
GROUP_EXACT = {"资产", "负债", "股东权益", "所有者权益", "负债及股东权益",
               "负债和股东权益", "流动资产", "非流动资产", "流动负债", "非流动负债"}
GROUP_COLON_RE = re.compile(r"^(?:[\u4e00-\u9fa5]{2,12})[：:]$")
# 子项前缀（仅 2 个数值，无附注号）
SUBITEM_PREFIX_RE = re.compile(r"^(?:其中|减|加)\s*[：:]")

# 文本流主表区间内的重印表头/元数据噪声
NOTE_HEADER_RE = re.compile(r"^附注\s*[一二三四五六七八九十]+$")
DATE_HEADER_RE = re.compile(r"^20\d{2}\s*年\s*(?:1[0-2]\s*月\s*3?[01]\s*日)?$")
HALF_DATE_RE = re.compile(r"^1[0-2]\s*月\s*3?[01]\s*日$")
YEAR_LINE_RE = re.compile(r"^20\d{2}\s*年度?$")
SCOPE_LINE_RE = re.compile(r"^(?:本集团|本公司)$")
UNIT_LINE_RE = re.compile(r"(除.*?注明外|金额单位|单位[：:])")
# 区间终止：签署语 / 签章
SIGN_OFF_RE = re.compile(r"(后附财务报表附注|此财务报表已于|财务报表附注为.*组成部分)")

# 跨页 prose 缝合：句读终止符集合
SENTENCE_END_CHARS = set("。！？；：…”\"’）)")

# 中文括号负数形如 (1,234.56)
PAREN_NEG_RE = re.compile(r"^\((-?[\d,]+(?:\.\d+)?)\)$")


def cn_num_to_int(s: str):
    # 仅处理 一..十二、二十 等年报常见序号；无法解析返回 None
    if not s:
        return None
    if s == "十":
        return 10
    if s.startswith("十"):
        return 10 + CN_DIGITS.get(s[1:], 0)
    if s.endswith("十"):
        return CN_DIGITS.get(s[0], 0) * 10
    if "十" in s:
        a, b = s.split("十", 1)
        return CN_DIGITS.get(a, 0) * 10 + CN_DIGITS.get(b, 0)
    return CN_DIGITS.get(s)


def is_number_token(s: str) -> bool:
    if s is None:
        return False
    s = s.strip()
    return bool(NUMBER_TOKEN_RE.match(s) or DASH_NULL_RE.match(s))


def normalize_number(s: str):
    # 返回去千分位的数值串；空值破折号返回 None；括号负数转 - 前缀
    if s is None:
        return None
    s = s.strip().replace(",", "").replace(" ", "")
    if not s or DASH_NULL_RE.match(s):
        return None
    m = PAREN_NEG_RE.match(s)
    if m:
        return "-" + m.group(1).replace(",", "")
    if NUMBER_TOKEN_RE.match(s):
        return s
    return None


def parse_statement(line: str):
    # 识别主表标题行，返回 (scope, kind, is_continuation) 或 None
    # scope: merge=合并表, parent=母公司/公司表, unknown=裸标题需靠上下文
    if not line:
        return None
    s = line.strip()
    is_cont = bool(CONTINUATION_RE.search(s))
    s = CONTINUATION_RE.sub("", s).strip()
    m = STATEMENT_LINE_RE.match(s)
    if not m:
        return None
    lead1, lead2, kind_text = m.group(1), m.group(2), m.group(3)
    kind = _STATEMENT_KINDS.get(kind_text)
    if not kind:
        return None
    lead = (lead1 or "") + (lead2 or "")
    if "合并" in lead:
        scope = "merge"
    elif lead1 in ("母公司", "公司"):
        scope = "parent"
    else:
        scope = "unknown"
    return scope, kind, is_cont


def is_tail_subject(text: str) -> bool:
    t = (text or "").strip()
    return any(k in t for k in TAIL_MARKERS)


def classify_subject(text: str):
    # 文本流科目行分类：group / subitem / item / None
    t = (text or "").strip().lstrip("#").strip()
    if not t:
        return None
    if t in GROUP_EXACT or GROUP_COLON_RE.match(t):
        return "group"
    if SUBITEM_PREFIX_RE.match(t):
        return "subitem"
    if is_tail_subject(t):
        return "tail"
    if SUBJECT_LINE_RE.match(t) and not UNIT_LINE_RE.search(t) and not NOTE_HEADER_RE.match(t):
        return "item"
    return None
