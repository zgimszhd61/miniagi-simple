#coding:utf-8
# 该模块提供了`MiniAGI`类，这是一个自主代理的实现，它与用户交互并执行任务，支持实时监控其行为、对其性能进行批评以及保留行动记忆。
import os
import sys
import re
import platform
import urllib
from pathlib import Path
from urllib.request import urlopen
from termcolor import colored
import openai
from thinkgpt.llm import ThinkGPT
import tiktoken
from bs4 import BeautifulSoup
from spinner import Spinner
from commands import Commands
from exceptions import InvalidLLMResponseError
import os
os.environ["OPENAI_API_KEY"] = "sk-"
operating_system = platform.platform()
PROMPT = f"你是在{operating_system}上运行的自主代理。" + '''
目标：{objective}（例如“查找巧克力曲奇食谱”）

你正在逐步朝着目标努力。之前的步骤：

{context}

你的任务是回应下一个动作。
支持的命令有：

命令 | 参数
-----------------------
memorize_thoughts | 内部辩论、细化、规划
execute_python | Python 代码（多行）
execute_shell | shell 命令（非交互式，单行）
ingest_data | 输入文件或 URL
process_data | 提示|输入文件或 URL
talk_to_user | 要说的话
done | 无

强制动作格式为：

<r>[YOUR_REASONING]</r><c>[COMMAND]</c>
[ARGUMENT]

ingest_data 和 process_data 不能处理多个文件/URL 参数。一次只能指定一个。
使用 process_data 处理具有更大上下文窗口的大量数据。
使用 execute_python 运行的 Python 代码必须以输出“print”语句结束。
不要搜索 GPT3/GPT4 已经知道的信息。
使用 memorize_thoughts 整理你的想法。
memorize_thoughts 参数不能为空！
如果目标已实现，请发送“done”命令。
回应时只能包含一个想法/命令/参数组合。
不要链接多个命令。
命令之前或之后不要有额外文本。
不要重复之前执行的命令。

每个动作都会返回一个观察结果。重要提示：观察结果可能会被总结以适应你有限的记忆。

示例动作：

<r>考虑可以转化为在线工作的技能和兴趣。</r><c>memorize_thoughts</c>
我有数据录入和分析以及社交媒体管理的经验。
(...)

<r>搜索具有巧克力曲奇食谱的网站。</r><c>web_search</c>
巧克力曲奇食谱

<r>摄入有关巧克力曲奇的信息。</r><c>ingest_data</c>
https://example.com/chocolate-chip-cookies

<r>阅读本地文件 /etc/hosts。</r><c>ingest_data</c>
/etc/hosts

<r>提取有关巧克力曲奇的信息。</r><c>process_data</c>
提取巧克力曲奇食谱|https://example.com/chocolate-chip-cookies

<r>总结这篇 Stackoverflow 文章。</r><c>process_data</c>
总结这篇文章的内容|https://stackoverflow.com/questions/1234/how-to-improve-my-chatgpt-prompts

<r>审查此代码以查找安全问题。</r><c>process_data</c>
审查此代码以查找安全漏洞|/path/to/code.sol

<r>我需要向用户寻求指导。</r><c>talk_to_user</c>
有关巧克力曲奇食谱的网站的 URL 是什么？

<r>将“Hello, world!”写入文件</r><c>execute_python</c>
with open('hello_world.txt', 'w') as f:
    f.write('Hello, world!')

<r>目标已完成。</r><c>done</c>
'''

RETRIEVAL_PROMPT = "你将被要求处理来自 URL 或文件的数据。你无需自己访问 URL 或文件，它将被加载并包含为“INPUT_DATA”。"

OBSERVATION_SUMMARY_HINT = "使用简短的句子和缩写总结文本。"

HISTORY_SUMMARY_HINT = "你是一个自主代理，正在总结你的历史。根据你的历史摘要和最新动作生成一个新摘要。包括所有先前动作的列表。保持简短。使用简短的句子和缩写。"

class MiniAGI:
    """
    代表一个自主代理。

    属性:
        agent: `ThinkGPT` 的一个实例，用于生成代理的动作。
        summarizer: `ThinkGPT` 的一个实例，用于生成代理历史的摘要。
        objective (str): 代理正在努力实现的目标。
        max_context_size (int): 代理短期记忆的最大大小（以标记计数）。
        max_memory_item_size (int): 记忆项的最大大小（以标记计数）。
        debug (bool): 指示是否打印调试信息。
        summarized_history (str): 代理动作的摘要历史。
        criticism (str): 代理最后一个动作的批评。
        thought (str): 代理最后一个动作的推理。
        proposed_command (str): 代理建议执行的下一个命令。
        proposed_arg (str): 建议命令的参数。
        encoding: 代理模型词汇表的编码。
    """

    def __init__(
        self,
        agent_model: str,
        summarizer_model: str,
        objective: str,
        max_context_size: int,
        max_memory_item_size: int,
        debug: bool = False
        ):
        """
        构造一个 `MiniAGI` 实例。

        参数:
            agent_model (str): 用作代理的模型名称。
            summarizer_model (str): 用于摘要的模型名称。
            objective (str): 代理的目标。
            max_context_size (int): 代理记忆的最大上下文大小（以标记计数）。
            max_memory_item_size (int): 记忆项的最大大小（以标记计数）。
            debug (bool, 可选): 一个标志，指示是否打印调试信息。
        """

        self.agent = ThinkGPT(
            model_name=agent_model,
            request_timeout=600,
            verbose=False
        )

        self.summarizer = ThinkGPT(
            model_name=summarizer_model,
            request_timeout=600,
            verbose=False
        )
        self.objective = objective
        self.max_context_size = max_context_size
        self.max_memory_item_size = max_memory_item_size
        self.debug = debug

        self.summarized_history = ""
        self.criticism = ""
        self.thought = ""
        self.proposed_command = ""
        self.proposed_arg = ""

        self.encoding = tiktoken.encoding_for_model(self.agent.model_name)

    # 更新代理的记忆，包括执行的动作和观察到的结果。
    # 可选地，还可以更新代理历史的摘要。
    def __update_memory(
            self,
            action: str,
            observation: str,
            update_summary: bool = True
        ):
        """
        更新代理的记忆，包括最后执行的动作和其观察结果。
        可选地，还可以更新代理历史的摘要。

        参数:
            action (str): ThinkGPT实例执行的动作。
            observation (str): ThinkGPT实例在执行动作后进行的观察。
            update_summary (bool, optional): 是否更新摘要。
        """

        # 如果观察结果的编码长度超过最大记忆项大小，则使用摘要器进行摘要。
        if len(self.encoding.encode(observation)) > self.max_memory_item_size:
            observation = self.summarizer.chunked_summarize(
                observation, self.max_memory_item_size,
                instruction_hint=OBSERVATION_SUMMARY_HINT
                )

        # 根据动作类型，构造新的记忆项。
        if "memorize_thoughts" in action:
            new_memory = f"ACTION:\nmemorize_thoughts\nTHOUGHTS:\n{observation}\n"
        else:
            new_memory = f"ACTION:\n{action}\nRESULT:\n{observation}\n"

        # 如果需要更新摘要，则将新的记忆项添加到摘要中。
        if update_summary:
            self.summarized_history = self.summarizer.summarize(
                f"Current summary:\n{self.summarized_history}\nAdd to summary:\n{new_memory}",
                self.max_memory_item_size,
                instruction_hint=HISTORY_SUMMARY_HINT
                )

        # 将新的记忆项添加到代理的记忆中。
        self.agent.memorize(new_memory)

    # 获取代理当前的上下文，用于思考和行动。
    def __get_context(self) -> str:
        """
        获取代理用于思考和行动的上下文。

        返回:
            str: 代理的上下文。
        """

        # 计算摘要和批评的编码长度。
        summary_len = len(self.encoding.encode(self.summarized_history))
        criticism_len = len(self.encoding.encode(self.criticism)) if len(self.criticism) > 0 else 0

        # 根据可用的上下文大小，从代理的记忆中回忆动作。
        action_buffer = "\n".join(
                self.agent.remember(
                limit=32,
                sort_by_order=True,
                max_tokens=self.max_context_size - summary_len - criticism_len
            )
        )

        # 构造并返回上下文字符串。
        return f"SUMMARY\n{self.summarized_history}\nPREV ACTIONS:"\
            f"\n{action_buffer}\n{self.criticism}"


    # 使用`ThinkGPT`模型预测代理应该采取的下一个行动。
    def think(self):
        """
        使用`ThinkGPT`模型预测代理应该采取的下一个行动。
        """

        context = self.__get_context()

        # if self.debug:
        #     print(context)
        # print("PROMPT-------")
        # print(PROMPT.format(context=context, objective=self.objective))

        response_text = self.agent.predict(
            prompt=PROMPT.format(context=context, objective=self.objective)
        )

        # if self.debug:
        #     print(f"RAW RESPONSE:\n{response_text}")

        PATTERN = r'^<r>(.*?)</r><c>(.*?)</c>\n*(.*)$'

        try:
            match = re.search(PATTERN, response_text, flags=re.DOTALL | re.MULTILINE)

            _thought = match[1]
            _command = match[2]
            _arg = match[3]
        except Exception as exc:
            raise InvalidLLMResponseError from exc

        # 移除不需要的代码格式化反引号
        _arg = _arg.replace("```", "")

        self.thought = _thought
        self.proposed_command = _command
        self.proposed_arg = _arg

    # 检索代理的最后一个思考、建议的命令和参数。
    def read_mind(self) -> tuple:
        """
        检索代理的最后一个思考、建议的命令和参数。

        返回:
            tuple: 包含代理的思考、建议的命令和参数的元组。
        """

        _arg = self.proposed_arg.replace("\n", "\\n") if len(self.proposed_arg) < 64\
            else f"{self.proposed_arg[:64]}...".replace("\n", "\\n")

        return (
            self.thought,
            self.proposed_command,
            _arg
        )

    # 从URL或文件中检索内容。
    @staticmethod
    def __get_url_or_file(_arg: str) -> str:
        """
        从URL或文件中检索内容。

        参数:
            arg (str): URL或文件名

        返回:
            str: 观察结果：URL或文件的内容。
        """

        if _arg.startswith("http://") or _arg.startswith("https://"):
            with urlopen(_arg) as response:
                html = response.read()
            data = BeautifulSoup(
                    html,
                    features="lxml"
                ).get_text()
        else:
            with open(_arg, "r") as file:
                data = file.read()

        return data

    # 处理来自URL或文件的数据。
    def __process_data(self, _arg: str) -> str:
        """
        处理来自URL或文件的数据。

        参数:
            arg (str): 提示和URL/文件名，用|分隔

        返回:
            str: 观察结果：处理URL或文件的结果。
        """
        args = _arg.split("|")

        if len(args) == 1:
            return "Invalid command. The correct format is: prompt|file or url"

        if len(args) > 2:
            return "Cannot process multiple input files or URLs. Process one at a time."

        (prompt, __arg) = args

        try:
            input_data = self.__get_url_or_file(__arg)
        except urllib.error.URLError as e:
            return f"Error: {str(e)}"
        except OSError as e:
            return f"Error: {str(e)}"

        if len(self.encoding.encode(input_data)) > self.max_context_size:
            input_data = self.summarizer.chunked_summarize(
                input_data, self.max_context_size,
                instruction_hint=OBSERVATION_SUMMARY_HINT
                )

        print(f"{RETRIEVAL_PROMPT}\n{prompt}\nINPUT DATA:\n{input_data}")
        
        return self.agent.predict(
                prompt=f"{RETRIEVAL_PROMPT}\n{prompt}\nINPUT DATA:\n{input_data}"
            )

    # 执行代理建议的命令并更新代理的记忆。
    def act(self):
        """
        执行代理建议的命令并更新代理的记忆。
        """
        if command == "process_data":
            obs = self.__process_data(self.proposed_arg)
        elif command == "ingest_data":
            obs = self.__ingest_data(self.proposed_arg)
        else:
            obs = Commands.execute_command(self.proposed_command, self.proposed_arg)

        self.__update_memory(f"{self.proposed_command}\n{self.proposed_arg}", obs)
        self.criticism = ""

    # 使用用户对代理最后行动的响应更新代理的记忆。
    def user_response(self, response):
        """
        使用用户对代理最后行动的响应更新代理的记忆。

        参数:
            response (str): 用户对代理最后行动的响应。
        """
        # print("user_response-----")
        # print(f"{self.proposed_command}\n{self.proposed_arg}", response)
        self.__update_memory(f"{self.proposed_command}\n{self.proposed_arg}", response)
        self.criticism = ""

# 当该脚本被直接运行时执行以下代码
if __name__ == "__main__":

    # 检查命令行参数数量，如果不等于2，则提示正确的使用方法并退出
    if len(sys.argv) != 2:
        print("Usage: main.py <objective>")
        sys.exit(0)

    # 从环境变量获取工作目录路径
    # 如果工作目录未设置或为空，则默认设置为用户主目录下的miniagi文件夹
    work_dir = os.getenv("WORK_DIR")
    if work_dir is None or not work_dir:
        work_dir = os.path.join(Path.home(), "miniagi")
        # 如果工作目录不存在，则创建该目录
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
    # 打印当前工作目录,尝试切换到工作目录，如果目录不存在，则提示并退出
    print(f"工作目录是 {work_dir}")
    try:
        os.chdir(work_dir)
    except FileNotFoundError:
        print("Directory doesn't exist. Set WORK_DIR to an existing directory or leave it blank.")
        sys.exit(0)

    # 初始化MiniAGI对象，传入模型、摘要模型、目标、上下文大小限制、记忆项大小限制和调试模式标志
    miniagi = MiniAGI(
        "gpt-4",
        "gpt-3.5-turbo",
        sys.argv[1],
        int(4000),
        int(2000),
        False
    )
    # 主循环(核心代码)
    while True:
        try:
            # 显示旋转指示器，表示正在处理
            with Spinner():
                miniagi.think()
        except InvalidLLMResponseError:
            # 如果收到无效的LLM响应，则打印错误信息并重试
            print(colored("LLM 响应无效，正在重试...", "red"))
            continue

        # 读取MiniAGI的思考结果(planning起作用)
        (thought, command, arg) = miniagi.read_mind()

        # 打印MiniAGI的思考结果、命令和参数
        print(colored(f"MiniAGI: {thought}\nCmd: {command}, Arg: {arg}", "cyan"))

        # 如果命令是"done"，则退出程序
        if command == "done":
            sys.exit(0)

        # 如果命令是"talk_to_user"，则与用户交互
        if command == "talk_to_user":
            print(colored(f"MiniAGI: {miniagi.proposed_arg}", 'blue'))
            user_input = input('Your response(if want to end,type done): ')
            ## 如果用户输入done,那么整个任务结束.
            if user_input == "done":
                print(colored("任务结束,合作愉快"))
                break
            with Spinner():
                miniagi.user_response(user_input)
            continue

        # 如果命令是"memorize_thoughts"，则打印MiniAGI正在思考的内容
        if command == "memorize_thoughts":
            print(colored("MiniAGI is thinking:\n"\
                f"{miniagi.proposed_arg}", 'cyan'))

        # 执行MiniAGI的行动
        with Spinner():
            miniagi.act()
