# 从巨潮资讯网下载 50 股 × 2 年的年报 PDF，
# 落到`data/raw/pdf/{code}/{year}_annual.pdf` ，并在`manifest.jsonl` 登记。
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    PROJECT_ROOT, Manifest, RateLimitedSession, load_config, load_env,
    market_of, now_iso, read_stock_list, setup_logging,
)

#巨潮搜索接口，用股票代码查机构 ID（orgId）
CNINFO_SEARCH = "https://www.cninfo.com.cn/new/information/topSearch/query" 
#巨潮的"年度报告"分类代号
ANNUAL_CATEGORY = "category_ndbg_szsh"
# 巨潮查询接口要求带 Referer，否则部分场景返回空结果
CNINFO_HEADERS = {
    "Referer": "https://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search",
}
# 非正式年报标题中的排除关键词
EXCLUDE_KEYWORDS = ("取消", "英文", "摘要", "补充", "公告", "征求意见")
# 修订版关键词，命中时优先选用（更正后版本更权威）
REVISE_KEYWORDS = ("修订", "更正", "更新")

def resolve_org_id(session, code: str):
    # 巨潮查询依赖 code,orgId 组合，先用代码搜出 orgId
    resp = session.post(CNINFO_SEARCH, data={"keyWord": code, "maxNum": 10})
    for item in resp.json():
        if item.get("code") == code:
            return item.get("orgId")
    return None


def pick_annual(anns: list, year: int):
    # 标题写法兼容：2024年年度报告 / 2023年度报告 /
    # 2024年年报报告（录入错误）/ 2024年年度报（标题截断，如大秦铁路）
    title_pattern = re.compile(
        rf"{year}(?:年度报告|年年度报告|年年报报告|年年度报)"
    )
    candidates = []
    for ann in anns:
        #去掉标题里的空格再匹配（巨潮标题有时带空格）
        title = (ann.get("announcementTitle") or "").replace(" ", "")
        if not ann.get("adjunctUrl"):
            continue
        #标题不匹配年份模式就跳过
        if not title_pattern.search(title):
            continue
        #标题含排除关键词（摘要/英文/取消等）就跳过。
        if any(kw in title for kw in EXCLUDE_KEYWORDS):
            continue
        candidates.append(ann)
    # 存在修订/更正版本时优先取修订版
    for ann in candidates:
        if any(kw in ann["announcementTitle"] for kw in REVISE_KEYWORDS):
            return ann
    # 同名多份时（如重新上传的更正件），取 adjunctUrl 编号最大者
    if candidates:
        return max(candidates, key=lambda a: int(re.sub(r"\D", "", a["adjunctUrl"]) or 0))
    return None


def query_annual_reports(session, code: str, org_id: str, year: int,
                         cfg: dict, category: str = ANNUAL_CATEGORY) -> list:
    # 年报在次年 1-4 月披露，用 seDate 锁定该窗口
    # category 传空串时不按公告分类过滤，用于年报被错挂分类时的兜底
    season = cfg["datasource"]["cninfo"]["report_season"]
    payload = {
        "pageNum": 1,
        "pageSize": cfg["datasource"]["cninfo"]["page_size"],
        "column": market_of(code),
        "tabName": "fulltext", 
        "stock": f"{code},{org_id}",
        "searchkey": "",
        "category": category, #默认是年报分类，传空串表示不按分类过滤（兜底用
        "seDate": f"{year + 1}-{season['start']}~{year + 1}-{season['end']}",
        "sortName": "",
        "sortType": "",
        "isHLtitle": "true",
    }
    resp = session.post(cfg["datasource"]["cninfo"]["query_url"], data=payload)
    return resp.json().get("announcements") or []


def main() -> None:
    #命令行参数：`--codes 600519 000333` 指定股票，
    # `--years 2024` 指定年份。不指定就用默认（股票池 + 配置里的年份）。
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes", nargs="*", help="指定代码，缺省读 stock_list.txt")
    parser.add_argument("--years", nargs="*", type=int, help="指定年份，缺省读 config")
    args = parser.parse_args()
    #初始化：加载环境变量、读配置、设日志。确定要处理的股票列表和年份列表
    load_env()
    cfg = load_config()
    log = setup_logging()

    ds = cfg["datasource"]
    codes = args.codes or read_stock_list(cfg["paths"]["stock_list"])
    years = args.years or ds["years"]

    #创建限流会话（参数从配置读）、登记表、PDF 存储根目录、巨潮 PDF 下载前缀
    session = RateLimitedSession(
        ds["request_interval_sec"], ds["max_retries"], ds["timeout"], CNINFO_HEADERS
    )
    manifest = Manifest(str(Path(cfg["paths"]["parsed_metadata"]) / "manifest.jsonl"))
    pdf_root = PROJECT_ROOT / cfg["paths"]["raw_pdf"]
    pdf_base = ds["cninfo"]["pdf_base"]
    #外层循环股票。先查 orgId，查不到就跳过并记警告。
    for code in codes:
        org_id = resolve_org_id(session, code)
        if not org_id:
            log.warning("未找到 orgId，跳过 %s", code)
            continue
        for year in years:
            doc_id = f"{code}_{year}_annual"
            if manifest.has(doc_id):
                log.info("已登记，跳过 %s", doc_id)
                continue
            #内层循环年份。`doc_id` 格式`600519_2023_annual` ，作为去重键。已登记就跳过（断点续传）
            anns = query_annual_reports(session, code, org_id, year, cfg)
            ann = pick_annual(anns, year)
            #两次都没找到，记警告跳过
            if not ann:
                # 兜底：个别公司年报被错挂公告分类（如大秦铁路 2024），
                # 不限分类在同一时间窗再查一次
                anns_fb = query_annual_reports(
                    session, code, org_id, year, cfg, category=""
                )
                ann = pick_annual(anns_fb, year)
            if not ann:
                log.warning("无正式年报 %s %s（主分类 %s 条，全分类 %s 条）",
                            code, year, len(anns), len(anns_fb))
                continue
            #拼出 PDF 下载地址并下载。`.content` 取二进制内容。 
            # 完整性校验 ：合法 PDF 文件以`%PDF-` 开头（4 个字节），
            # 不是就说明下载失败或被重定向到错误页面，跳过。
            pdf_url = f"{pdf_base}/{ann['adjunctUrl']}"
            content = session.get(pdf_url).content
            # PDF 完整性校验：合法文件以 %PDF- 开头
            if not content.startswith(b"%PDF-"):
                log.error("下载内容非 PDF，跳过 %s（%s 字节）", doc_id, len(content))
                continue
            #确定存储路径`data/raw/pdf/600519/2024_annual.pdf` ，建目录，写二进制文件
            dest = pdf_root / code / f"{year}_annual.pdf"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)
            #登记到 manifest，记录血缘信息：文档 ID、股票、年份、标题、
            # 本地路径（相对路径、正斜杠跨平台）、源 URL、文件大小、下载时间
            manifest.append({
                "doc_id": doc_id,
                "stock_code": code,
                "year": year,
                "type": "annual",
                "title": ann.get("announcementTitle"),
                "pdf_path": str(dest.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "source_url": pdf_url,
                "size_bytes": len(content),
                "downloaded_at": now_iso(),
            })
            log.info("已下载 %s | %s | %s bytes",
                     doc_id, ann.get("announcementTitle"), len(content))


if __name__ == "__main__":
    main()
