# 导入必要的库
import sys
import time
import threading

class Spinner:
    """
    实现了一个旋转光标效果
    """
    busy = False
    delay = 0.1

    @staticmethod
    def spinning_cursor():
        """
        生成器，产生用于旋转动画的光标字符序列
        
        Yields:
            str: 用于旋转动画的光标字符
        """
        while 1:
            for cursor in '|/-\\':
                yield cursor

    def __init__(self, delay=None):
        """
        初始化一个 Spinner 对象，可选自定义光标帧之间的延迟
            
        Args:
            delay (float, optional): 光标帧之间的延迟时间，单位为秒。默认为 None。
        """
        self.spinner_generator = self.spinning_cursor()
        if delay and float(delay):
            self.delay = delay

    def spinner_task(self):
        """
        旋转光标动画任务。将光标字符写入 stdout，刷新，延迟指定时间，然后回退光标。
        """
        while self.busy:
            sys.stdout.write(next(self.spinner_generator))
            sys.stdout.flush()
            time.sleep(self.delay)
            sys.stdout.write('\b')
            sys.stdout.flush()

    def __enter__(self):
        self.busy = True
        threading.Thread(target=self.spinner_task).start()

    def __exit__(self, exception, value, tb):
        self.busy = False
        time.sleep(self.delay)
        if exception is not None:
            return False
        return True