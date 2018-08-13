"""
https://github.com/mlouielu/twstock
"""

"""
https://www.finlab.tw/%E8%B6%85%E7%B0%A1%E5%96%AE%E5%8F%B0%E8%82%A1%E6%AF%8F%E6%97%A5%E7%88%AC%E8%9F%B2%E6%95%99%E5%AD%B8/
"""
import requests
from io import StringIO
import pandas as pd
import numpy as np


datestr = '20180131'
r = requests.post('http://www.twse.com.tw/exchangeReport/MI_INDEX?response=csv&date=' + datestr + '&type=ALL')
df = pd.read_csv(StringIO("\n".join([i.translate({ord(c): None for c in ' '}) 
for i in r.text.split('\n') 
    if len(i.split('",')) == 17 and i[0] != '='])), header=0)

	
df[pd.to_numeric(df['本益比'], errors='coerce') < 15]
