import numpy as np
from typing import Callable, List


def square(x: ndarray) -> ndarray:
    '''
    将输入 ndarray 中的每个元素进行平方计算
    '''
    return np.power(x, 2)


def leaky_relu(x: ndarray) -> ndarray:
    '''
    将 Leaky ReLU 函数应用于 ndarray 中的每个元素
    '''
    return np.maximum(0.2 * x, x)


def sigmoid(x: ndarray) -> ndarray:
    '''
    将 sigmoid 函数应用于输入 ndarray 中的每个元素
    '''
    return 1 / (1 + np.exp(-x))


def deriv(func: Callable[[ndarray], ndarray], input_: ndarray, delta: float = 0.01) -> ndarray:
    '''
    计算函数 func 在 input_ 数组中的每个元素处的导数
    '''
    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)

# 为嵌套函数定义一个数据类型
Array_Function = Callable[[ndarray], ndarray]
Chain = List(Array_Function)

def chain_length_2(chain: Chain, a: ndarray) -> ndarray:
    '''
    在一行代码中计算“链”中的两个函数
    '''
    assert len(chain) == 2, "Length of input 'chain' should be 2"
    f1 = chain[0]
    f2 = chain[1]
    return f2(f1)


def chain_deriv_2(chain: Chain, input_shape: ndarray) -> ndarray:
    ''''
    使用链式法则计算两个嵌套函数的导数: (f2f1(x))' = f2'(f1(x))*f1'(x)
    '''
    assert len(chain) == 2, "This function requires 'Chain' objects of length 2"
    assert input_range.ndim == 1, "Function requires a 1 dimensional ndarray as input_range"
    f1 = chain[0]
    f2 = chain[1]
    # df1/dx
    f1_of_x = f1(input_range)
    # df1/du
    df1dx = deriv(f1, input_range)
    # df2/du(f1(x))
    df2du = deriv(f2, f1(input_range))
    # 在每一点上将这些量相乘
    return df1x * df2du


if __name__ == "__main__":
    PLOT_RANGE = np.arange(-3, 3, 0.01)
    chain_1 = [square, sigmoid]
    chain_2 = [sigmoid, square]
    plot_chain(chain_1, PLOT_RANGE)
    plot_chain_deriv(chain_1, PLOT_RANGE)
    plot_chain(chain_2, PLOT_RANGE)
    plot_chain_deriv(chain_2, PLOT_RANGE)
    
