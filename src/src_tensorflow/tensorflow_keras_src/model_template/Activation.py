import numpy as np


class Activation(object):
    """
    自定义激活函数
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def step_function(self, x):
        """
        阶跃(Step)函数
        """
        y = np.array(x > 0, dtype = np.int)
        return y

    def sigmoid(self x):
        """
        Sigmod 函数
        """
        y = 1 / (1 + np.exp(-x))
        return y

    def relu(self, x):
        """
        整流线性单元函数
        """
        y = np.maximum(0, x)
        return y

    def identity_function(x):
        """
        恒等函数
        """
        y = x
        return y

    def softmax(x):
        """
        Softmax 函数
        """
        exp_a = np.exp(x)
        sum_exp_x = np.sum(exp_x)
        y = exp_x / sum_exp_x
        return y


if __name__ == "__main__":
    pass