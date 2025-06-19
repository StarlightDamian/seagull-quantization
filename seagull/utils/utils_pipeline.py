# -*- coding: utf-8 -*-
"""
@Date: 2025/5/20 13:07
@Author: Damian
@Email: zengyuwei1995@163.com
@File: utils_pipeline.py
@Description: 工作流
"""


class Data:
    def __call__(self, *input, **kwargs):
        # 执行一些额外逻辑，比如 hooks 之类的
        return self.pipeline(*input, **kwargs)
    
    def pipeline(self, *input):
        ...


if __name__ == '__main__':
    class FeedForward(Data):
        def __init__(self, dropout=0.1):
            """
            d_model: 模型维度，d_ff: 前馈网络中间层维度
            """
            super().__init__()
    
        def pipeline(self, x):
            print(4)
        
    model = FeedForward()
    output = model(1)
