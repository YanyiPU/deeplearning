
TensorFLow Serving
=====================


1.TensorFLow Serving 安装
-------------------------------







2.TensorFLow Serving 模型部署
-------------------------------



3.在客户端调用以 TensorFLow  Serving 部署的模型
------------------------------------------------

TensorFLow Serving 支持使用 gRPC 方法和 RESTful API 方法调用以 TensorFLow Serving 部署的模型。

RESTful API 以标准的 HTTP POST 方法进行交互，请求和回复均为 JSON 对象。为了调用服务器端的模型，在客户端向服务器发送以下格式的请求.

    - 服务器 URI: ``http://服务器地址:端口号/v1/models/模型名:predict``

    - 请求内容

        .. code-block:: json

            {
                "signature_name": "需要调用的函数签名(Sequential模式不需要)",
                "instances": "输入数据"
            }

    - 回复:

        .. code-block:: json

            {
                "predictions": "返回值"
            }