from wttr_in import get_weather         # 注意：文件名改成了 wttr_in.py
from search_attraction import get_attraction  # 或者 search_attraction

# 将所有工具函数放入一个字典，方便后续调用
available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}
