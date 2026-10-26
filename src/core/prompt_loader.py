# 提示词加载：prompts/<name>.md 三段式（# 标题 / ## system / ## user）
# 提示词与业务代码解耦，统一存放 prompts/ 目录；本模块只做读取与切分，不解析变量
from dataclasses import dataclass

from core.paths import PROMPTS_DIR

# 三段式段落标记
_TITLE_MARK = "# "
_SYSTEM_MARK = "## system"
_USER_MARK = "## user"


@dataclass(frozen=True)
class Prompt:
    # 一份完整提示词
    title: str   # 标题段：提示词用途说明，便于人工检索
    system: str  # system 段：角色设定与行为约束
    user: str    # user 段：任务指令模板


def load_prompt(name: str) -> Prompt:
    # 读取 prompts/{name}.md 并切分为三段
    # name: 提示词文件名，不含 .md 后缀，如 "orchestrator_intent"
    path = PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"提示词文件不存在：{path}")

    title = ""
    system_lines: list[str] = []
    user_lines: list[str] = []
    # 当前所处段落：None（标题前）/ system / user
    section = None

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if section is None and stripped.startswith(_TITLE_MARK):
            # 标题段只认首个一级标题；进入正文段后不再识别 # 行
            title = stripped[len(_TITLE_MARK):].strip()
            continue
        if stripped == _SYSTEM_MARK:
            section = "system"
            continue
        if stripped == _USER_MARK:
            section = "user"
            continue
        # 按当前段落归集原文行，保持缩进与换行原样
        if section == "system":
            system_lines.append(line)
        elif section == "user":
            user_lines.append(line)

    if not title:
        raise ValueError(f"提示词缺少一级标题段（# 标题）：{path}")
    if not system_lines:
        raise ValueError(f"提示词缺少 ## system 段：{path}")
    if not user_lines:
        raise ValueError(f"提示词缺少 ## user 段：{path}")

    return Prompt(
        title=title,
        system="\n".join(system_lines).strip(),
        user="\n".join(user_lines).strip(),
    )
