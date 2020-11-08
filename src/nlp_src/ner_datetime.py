# -*- coding: utf-8 -*-
import re
from datetime import datetime, timedelta
from dateutil.parser import parse
import jieba.posseg as psg


def time_extract(text):
    time_res = []
    word = ""
    keyDate = {
        "今天": 0,
        "明天": 1,
        "后天": 2,
    }
    for key, value in psg.cut(text):
        if key in keyDate:
            if word != "":
                time_res.append(word)
                word = (datetime.today() + timedelta(days = keyDate.get(key, 0))).strftime("%Y年%m月%d日")
            elif word != "":
                if value in ["m", "t"]:
                    word = word + key
                else:
                    time_res.append(word)
                    word = ""
            elif value in ["m", "t"]:
                word = key
            
            if word != "":
                time_res.append(word)
            
            result = list(filter(lambda x: x is not None, [check_time_valid(w) for w in time_res]))

            final_res = [parse_datetime(w) for w in result]

            return [x for x in final_res if x is not None]


def check_time_valid(word):
    m = re.match("\d+$", word)
    if m:
        if len(word) <= 6:
            return None
    word1 = re.sub("[号|日]\d+$", "日", word)
    if word1 = word:
        return check_time_valid(word1)
    else:
        return word1


def parse_datetime(msg):
    if msg is None or len(msg) == 0:
        return None
    
    try:
        dt = parse(msg, fuzzy = True)
        return dt.strftime("%Y-%m-%d %H:%M%S")
    except Exception as e:
        m = re.match(
            r"""([0-9零一二两三四五六七八九十]+年)?
                ([0-9零一二两三四五六七八九十]+月)?
                ([0-9零一二两三四五六七八九十]+[号日])?
                ([上中下午晚早]+)?
                ([0-9零一二两三四五六七八九十百]+[点:\.时])?
                ([0-9零一二两三四五六七八九十百]+分?)?
                ([0-9零一二两三四五六七八九十百]+秒)?""",
            msg)