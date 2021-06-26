# -*- coding: utf-8 -*-
import opencc

converter = opencc.OpenCC("jp2t.json")
data = u'Open Chinese Convert（OpenCC）是一個開源的中文簡繁轉換項目，致力於製作高質量的基於統計預料的簡繁轉換詞庫。還提供函數庫(libopencc)、命令行簡繁轉換工具、人工校對工具、詞典生成程序、在線轉換服務及圖形用戶界面'
data2 = "プレバト"
data_new = converter.convert(data2)
print(data_new)
