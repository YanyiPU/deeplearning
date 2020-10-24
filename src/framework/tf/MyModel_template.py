import tensorflow as tf


class MyModel(tf.keras.Model):
    def __init__(self):
        super.__init__()
        # 此处添加初始化的代码(包含call方法中会用到的层)例如：
        layer1 = tf.keras.layers.BuildInLayer()
        layer2 = MyCustomLayer(...)

    def call(self, input):
        # 此处添加模型调用的代码(处理输入并返回输出)，例如：
        x = layer1(input)
        output = layer2(x)
        return output

if __name__ == "__main__":
    pass