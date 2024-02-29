"""
这个模块提供了一组可以执行不同命令的静态方法。
"""

import subprocess
from io import StringIO
from contextlib import redirect_stdout

# 禁用 pylint 的特定警告
# pylint: disable=broad-exception-caught, exec-used, unspecified-encoding

class Commands:
    """
    一组可以执行不同命令的静态方法。
    """

    @staticmethod
    def execute_command(command, arg) -> str:
        """
        执行与提供的命令字符串对应的命令。

        Args:
            command (str): 表示要执行的命令的命令字符串。
            arg (str): 要传递给命令的参数。

        Returns:
            str: 命令执行的结果，或者在执行过程中引发异常时的错误消息。
        """
        try:
            match command:
                case "memorize_thoughts":
                    result = Commands.memorize_thoughts(arg)
                case "execute_python":
                    result = Commands.execute_python(arg)
                case "execute_shell":
                    result = Commands.execute_shell(arg)
                case _:
                    result = f"未知命令: {command}"
        except Exception as exception:
            result = f"命令返回错误:\n{str(exception)}"

        return result

    @staticmethod
    def memorize_thoughts(arg: str) -> str:
        """
        简单地返回输入字符串。用于“记忆”一个想法。

        Args:
            arg (str): 输入字符串。

        Returns:
            str: 输入字符串。
        """
        return arg

    @staticmethod
    def execute_python(arg: str) -> str:
        """
        执行输入的 Python 代码并返回 stdout。

        Args:
            arg (str): 输入的 Python 代码。

        Returns:
            str: 执行的 Python 代码产生的 stdout。
        """
        _stdout = StringIO()
        with redirect_stdout(_stdout):
            exec(arg)

        return _stdout.getvalue()

    @staticmethod
    def execute_shell(arg: str) -> str:
        """
        执行输入的 shell 命令并返回 stdout 和 stderr。

        Args:
            arg (str): 输入的 shell 命令。

        Returns:
            str: 执行的 shell 命令产生的 stdout 和 stderr。
        """
        result = subprocess.run(arg, capture_output=True, shell=True, check=False)

        stdout = result.stdout.decode("utf-8")
        stderr = result.stderr.decode("utf-8")

        return f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"

