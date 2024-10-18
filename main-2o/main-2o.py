import os, json, traceback, subprocess, sys
from time import sleep
from litellm import completion

# ANSI转义码用于控制终端中的颜色和格式
class Colors:
    HEADER = '\033[95m'; OKBLUE = '\033[94m'; OKCYAN = '\033[96m'; OKGREEN = '\033[92m'
    WARNING = '\033[93m'; FAIL = '\033[91m'; ENDC = '\033[0m'; BOLD = '\033[1m'; UNDERLINE = '\033[4m'

# 配置部分
MODEL_NAME = os.environ.get('LITELLM_MODEL', 'anthropic/claude-3-5-sonnet-20240620')  # 从环境变量中获取模型名称，默认使用Claude-3-5
tools, available_functions = [], {}  # 工具列表和可用函数的字典
MAX_TOOL_OUTPUT_LENGTH = 5000  # 最大工具输出长度，必要时可以调整

# 自动检测可用的API密钥
api_key_patterns = ['API_KEY', 'ACCESS_TOKEN', 'SECRET_KEY', 'TOKEN', 'APISECRET']  # 常见API密钥的模式
available_api_keys = [key for key in os.environ.keys() if any(pattern in key.upper() for pattern in api_key_patterns)]  # 检查环境变量中的API密钥

# 注册工具函数，供后续使用
def register_tool(name, func, description, parameters):
    global tools
    tools = [tool for tool in tools if tool["function"]["name"] != name]  # 更新已有工具
    available_functions[name] = func  # 保存可用函数
    tools.append({
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": list(parameters.keys())
            }
        }
    })
    print(f"{Colors.OKGREEN}{Colors.BOLD}已注册工具:{Colors.ENDC} {name}")

# 创建或更新工具的函数
def create_or_update_tool(name, code, description, parameters):
    try:
        exec(code, globals())  # 动态执行代码
        register_tool(name, globals()[name], description, parameters)  # 注册新工具
        return f"工具 '{name}' 创建/更新成功。"
    except Exception as e:
        return f"创建/更新工具 '{name}' 时出错: {e}"

# 安装Python包
def install_package(package_name):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])  # 调用pip安装包
        return f"包 '{package_name}' 安装成功。"
    except Exception as e:
        return f"安装包 '{package_name}' 时出错: {e}"

# 序列化工具返回的结果，控制结果长度
def serialize_tool_result(tool_result, max_length=MAX_TOOL_OUTPUT_LENGTH):
    try:
        serialized_result = json.dumps(tool_result)  # 尝试将结果转换为JSON格式
    except TypeError:
        serialized_result = str(tool_result)  # 如果JSON转换失败，转换为字符串
    if len(serialized_result) > max_length:  # 检查结果是否超过最大长度
        return serialized_result[:max_length] + f"\n\n{Colors.WARNING}(注意: 结果已被截断至 {max_length} 字符。){Colors.ENDC}"
    else:
        return serialized_result

# 调用已注册的工具
def call_tool(function_name, args):
    func = available_functions.get(function_name)
    if not func:
        print(f"{Colors.FAIL}{Colors.BOLD}错误:{Colors.ENDC} 工具 '{function_name}' 未找到。")
        return f"工具 '{function_name}' 未找到。"
    try:
        print(f"{Colors.OKBLUE}{Colors.BOLD}调用工具:{Colors.ENDC} {function_name}，参数: {args}")
        result = func(**args)  # 调用工具并传递参数
        print(f"{Colors.OKCYAN}{Colors.BOLD}{function_name} 的结果:{Colors.ENDC} {result}")
        return result
    except Exception as e:
        print(f"{Colors.FAIL}{Colors.BOLD}错误:{Colors.ENDC} 执行 '{function_name}' 时出错: {e}")
        return f"执行 '{function_name}' 时出错: {e}"

# 标记任务完成
def task_completed():
    return "任务已标记为完成。"

# 初始化基础工具
register_tool("create_or_update_tool", create_or_update_tool, "根据指定名称、代码、描述和参数创建或更新工具。", {
    "name": {"type": "string", "description": "工具名称。"},
    "code": {"type": "string", "description": "工具的Python代码。"},
    "description": {"type": "string", "description": "工具的描述。"},
    "parameters": {
        "type": "object",
        "description": "定义工具参数的字典。",
        "additionalProperties": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "description": "参数的数据类型。"},
                "description": {"type": "string", "description": "参数的描述。"}
            },
            "required": ["type", "description"]
        }
    }
})

register_tool("install_package", install_package, "使用pip安装Python包。", {
    "package_name": {"type": "string", "description": "要安装的包名称。"}
})

register_tool("task_completed", task_completed, "标记当前任务为完成。", {})

# 主循环处理用户输入和LLM交互
def run_main_loop(user_input):
    # 将可用的API密钥包含在系统提示中
    if available_api_keys:
        api_keys_info = "可用的API密钥:\n" + "\n".join(f"- {key}" for key in available_api_keys) + "\n\n"
    else:
        api_keys_info = "没有可用的API密钥。\n\n"

    # 设置初始消息
    messages = [{
        "role": "system",
        "content": (
            "你是一个AI助手，设计用于迭代地构建和执行Python函数。"
            "你的任务是通过创建和使用工具循环执行请求的任务，直到任务完成。"
            "除非绝对必要，否则不要请求用户输入。"
            f"你有以下工具可用:\n\n{api_keys_info}"
        )
    }, {"role": "user", "content": user_input}]

    iteration, max_iterations = 0, 50  # 最大迭代次数
    while iteration < max_iterations:
        print(f"{Colors.HEADER}{Colors.BOLD}第 {iteration + 1} 次迭代中...{Colors.ENDC}")
        try:
            # 调用LLM的completion接口，获取响应
            response = completion(model=MODEL_NAME, messages=messages, tools=tools, tool_choice="auto")
            response_message = response.choices[0].message  # 解析LLM的返回结果
            if response_message.content:
                print(f"{Colors.OKCYAN}{Colors.BOLD}LLM响应:{Colors.ENDC}\n{response_message.content}\n")
            messages.append(response_message)
            # 如果有工具调用，执行工具
            if response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    tool_result = call_tool(function_name, args)
                    serialized_tool_result = serialize_tool_result(tool_result)
                    messages.append({
                        "role": "tool",
                        "name": function_name,
                        "tool_call_id": tool_call.id,
                        "content": serialized_tool_result
                    })
                if 'task_completed' in [tc.function.name for tc in response_message.tool_calls]:
                    print(f"{Colors.OKGREEN}{Colors.BOLD}任务完成。{Colors.ENDC}")
                    break
        except Exception as e:
            print(f"{Colors.FAIL}{Colors.BOLD}错误:{Colors.ENDC} 主循环中出错: {e}")
            traceback.print_exc()
        iteration += 1
        sleep(2)
    print(f"{Colors.WARNING}{Colors.BOLD}达到最大迭代次数或任务已完成。{Colors.ENDC}")

if __name__ == "__main__":
    run_main_loop(input(f"{Colors.BOLD}描述你想完成的任务: {Colors.ENDC}"))
