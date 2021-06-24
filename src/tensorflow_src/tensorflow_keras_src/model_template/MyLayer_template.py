import numpy as np
import tensorflow as tf

class MyLayer(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        # 初始化代码
    
    def build(self, input_shape): # input_shape 是一个 TensorShape 类型对象，提供输入的形状
        # 在第一次使用该层的时候调用该部分代码，在这里创建变量可以使得变量的形状自适应输入的形状
        # 而不需要使用者额外指定变量形状
        # 如果已经可以完全确定变量的形状，也可以在 __init__ 部分创建变量
        self.variable_0 = self.add_weight(...)
        self.variable_1 = self.add_weight(...)
    
    def call(self, inputs):
        # 模型调用的代码(处理输入并返回输出)
        return output
