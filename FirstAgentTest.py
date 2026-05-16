import os
import re
import sys
from pathlib import Path

from OpenAICompatibleClient import OpenAICompatibleClient
from system_prompt import AGENT_SYSTEM_PROMPT
from available_tools import available_tools


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(
            f"缺少环境变量 {name}。请复制 .env.example 为 .env 并填入密钥后重试。"
        )
    return value


_root = Path(__file__).resolve().parent
_load_env_file(_root / ".env")


def _env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


# --- 1. 配置LLM客户端 ---
API_KEY = _env("OPENAI_API_KEY", "API_KEY")
if not API_KEY:
    raise SystemExit(
        "缺少 OPENAI_API_KEY（或旧版 API_KEY）。请复制 .env.example 为 .env 并填入密钥。"
    )
BASE_URL = _env("OPENAI_BASE_URL", "BASE_URL") or "https://api-inference.modelscope.cn/v1/"
MODEL_ID = _env("MODEL_NAME", "MODEL_ID") or "deepseek-ai/DeepSeek-V3.2"
if not _env("TAVILY_API_KEY"):
    raise SystemExit("缺少 TAVILY_API_KEY。请在 .env 中配置。")

llm = OpenAICompatibleClient(
    model=MODEL_ID,
    api_key=API_KEY,
    base_url=BASE_URL
)


def get_destination() -> str:
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:]).strip()
    while True:
        destination = input("请输入目的地（城市名）: ").strip()
        if destination:
            return destination
        print("目的地不能为空，请重新输入。")


def build_user_prompt(destination: str) -> str:
    return (
        f"你好，请帮我查询一下今天{destination}的天气，"
        f"然后根据天气推荐一个合适的旅游景点。"
    )


def run_agent(user_prompt: str) -> None:
    prompt_history = [f"用户请求: {user_prompt}"]
    print(f"用户输入: {user_prompt}\n" + "=" * 40)

    for i in range(5):
        print(f"--- 循环 {i + 1} ---\n")
        full_prompt = "\n".join(prompt_history)

        llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)
        match = re.search(
            r"(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)",
            llm_output,
            re.DOTALL,
        )
        if match:
            truncated = match.group(1).strip()
            if truncated != llm_output.strip():
                llm_output = truncated
                print("已截断多余的 Thought-Action 对")
        print(f"模型输出:\n{llm_output}\n")
        prompt_history.append(llm_output)

        action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
        if not action_match:
            observation = (
                "错误: 未能解析到 Action 字段。"
                "请确保你的回复严格遵循 'Thought: ... Action: ...' 的格式。"
            )
            observation_str = f"Observation: {observation}"
            print(f"{observation_str}\n" + "=" * 40)
            prompt_history.append(observation_str)
            continue
        action_str = action_match.group(1).strip()

        if action_str.startswith("Finish"):
            final_answer = re.match(r"Finish\[(.*)\]", action_str).group(1)
            print(f"任务完成，最终答案: {final_answer}")
            break

        tool_name = re.search(r"(\w+)\(", action_str).group(1)
        args_str = re.search(r"\((.*)\)", action_str).group(1)
        kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))

        if tool_name in available_tools:
            observation = available_tools[tool_name](**kwargs)
        else:
            observation = f"错误:未定义的工具 '{tool_name}'"

        observation_str = f"Observation: {observation}"
        print(f"{observation_str}\n" + "=" * 40)
        prompt_history.append(observation_str)


if __name__ == "__main__":
    destination = get_destination()
    run_agent(build_user_prompt(destination))
