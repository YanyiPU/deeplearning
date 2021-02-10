from bert_serving.client import BertClient



bc = BertClient(ip = "localhost", check_version = False, check_length = False)
vec = bc.encode(["学习"])
print(vec)