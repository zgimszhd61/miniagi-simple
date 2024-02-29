# 该模块包含了特定于MiniAGI的异常。

class InvalidLLMResponseError(Exception):
    """当无法解析LLM响应时引发的异常。
    
    属性:
        无
    """
