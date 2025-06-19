# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 22:05:35 2025

@author: awei
"""
import efinance as ef
df=ef.stock.get_history_bill('300750')
print(df)
# =============================================================================
# import adata
# df = adata.stock.market.get_capital_flow(stock_code='300239',
#                                                                #start_date='2020-01-01'
#                                                                )
# print(df)
# =============================================================================
# =============================================================================
# import adata
# df = adata.stock.market.get_capital_flow(stock_code='000001',start_date='2024-10-01')
# print(df)
# =============================================================================


import requests


url="https://finance.pae.baidu.com/vapi/v1/fundsortlist?code=000001&market=ab&finance_type=stock&tab=day&from=history&date=20250101&pn=0&rn=20&finClientType=pc"

#data_list = res.json()["Result"]["content"]
#rusult={"ResultCode":0,"ResultNum":0,"QueryID":"15231164892049196913","Result":{"content":[{"closepx":"11.70","date":"2024/12/31","extMainIn":"-1.75亿","largeNetIn":"-7772.69万","littleNetIn":"+7135.96万","mediumNetIn":"+1.04亿","ratio":"-2.09%","showtime":"2024-12-31","superNetIn":"-9720.00万","time":"20241231"},{"closepx":"11.95","date":"2024/12/30","extMainIn":"+9233.50万","largeNetIn":"-5450.29万","littleNetIn":"+1342.10万","mediumNetIn":"-1.06亿","ratio":"+1.01%","showtime":"2024-12-30","superNetIn":"+1.47亿","time":"20241230"},{"closepx":"11.83","date":"2024/12/27","extMainIn":"-2.10亿","largeNetIn":"-4588.74万","littleNetIn":"+1.11亿","mediumNetIn":"+9825.19万","ratio":"-0.25%","showtime":"2024-12-27","superNetIn":"-1.64亿","time":"20241227"},{"closepx":"11.86","date":"2024/12/26","extMainIn":"-1319.21万","largeNetIn":"+6196.98万","littleNetIn":"+3169.22万","mediumNetIn":"-1850.01万","ratio":"-0.50%","showtime":"2024-12-26","superNetIn":"-7516.18万","time":"20241226"},{"closepx":"11.92","date":"2024/12/25","extMainIn":"+1910.36万","largeNetIn":"-1977.93万","littleNetIn":"-683.98万","mediumNetIn":"-1226.38万","ratio":"+0.51%","showtime":"2024-12-25","superNetIn":"+3888.29万","time":"20241225"},{"closepx":"11.86","date":"2024/12/24","extMainIn":"+5182.89万","largeNetIn":"-1.04亿","littleNetIn":"-236.17万","mediumNetIn":"-4946.72万","ratio":"+1.11%","showtime":"2024-12-24","superNetIn":"+1.56亿","time":"20241224"},{"closepx":"11.73","date":"2024/12/23","extMainIn":"+1.19亿","largeNetIn":"+3323.19万","littleNetIn":"-7438.79万","mediumNetIn":"-4440.74万","ratio":"+0.95%","showtime":"2024-12-23","superNetIn":"+8556.34万","time":"20241223"},{"closepx":"11.62","date":"2024/12/20","extMainIn":"+5295.59万","largeNetIn":"+1788.58万","littleNetIn":"-1562.41万","mediumNetIn":"-3733.19万","ratio":"+0.26%","showtime":"2024-12-20","superNetIn":"+3507.01万","time":"20241220"},{"closepx":"11.59","date":"2024/12/19","extMainIn":"-6495.94万","largeNetIn":"-4770.96万","littleNetIn":"+5109.94万","mediumNetIn":"+1386.00万","ratio":"-0.52%","showtime":"2024-12-19","superNetIn":"-1724.98万","time":"20241219"},{"closepx":"11.65","date":"2024/12/18","extMainIn":"+9105.16万","largeNetIn":"+496.35万","littleNetIn":"-3269.78万","mediumNetIn":"-5835.38万","ratio":"+1.04%","showtime":"2024-12-18","superNetIn":"+8608.81万","time":"20241218"},{"closepx":"11.53","date":"2024/12/17","extMainIn":"+1554.41万","largeNetIn":"-1249.65万","littleNetIn":"-1628.48万","mediumNetIn":"+74.07万","ratio":"-0.35%","showtime":"2024-12-17","superNetIn":"+2804.06万","time":"20241217"},{"closepx":"11.57","date":"2024/12/16","extMainIn":"-1502.87万","largeNetIn":"-300.82万","littleNetIn":"+973.83万","mediumNetIn":"+529.04万","ratio":"+0.09%","showtime":"2024-12-16","superNetIn":"-1202.05万","time":"20241216"},{"closepx":"11.56","date":"2024/12/13","extMainIn":"-2.44亿","largeNetIn":"-1272.32万","littleNetIn":"+1.05亿","mediumNetIn":"+1.39亿","ratio":"-2.45%","showtime":"2024-12-13","superNetIn":"-2.31亿","time":"20241213"},{"closepx":"11.85","date":"2024/12/12","extMainIn":"+4388.25万","largeNetIn":"+2640.95万","littleNetIn":"+1770.69万","mediumNetIn":"-6158.94万","ratio":"+1.02%","showtime":"2024-12-12","superNetIn":"+1747.31万","time":"20241212"},{"closepx":"11.73","date":"2024/12/11","extMainIn":"-1.14亿","largeNetIn":"+5819.53万","littleNetIn":"+8701.17万","mediumNetIn":"+2736.44万","ratio":"-0.51%","showtime":"2024-12-11","superNetIn":"-1.73亿","time":"20241211"},{"closepx":"11.79","date":"2024/12/10","extMainIn":"+1.80亿","largeNetIn":"+1.02亿","littleNetIn":"-4623.89万","mediumNetIn":"-1.34亿","ratio":"+1.03%","showtime":"2024-12-10","superNetIn":"+7849.92万","time":"20241210"},{"closepx":"11.67","date":"2024/12/09","extMainIn":"-1923.72万","largeNetIn":"+1719.26万","littleNetIn":"-227.84万","mediumNetIn":"+2151.56万","ratio":"+0.09%","showtime":"2024-12-09","superNetIn":"-3642.98万","time":"20241209"},{"closepx":"11.66","date":"2024/12/06","extMainIn":"+2.95亿","largeNetIn":"+780.10万","littleNetIn":"-9818.70万","mediumNetIn":"-1.97亿","ratio":"+1.92%","showtime":"2024-12-06","superNetIn":"+2.87亿","time":"20241206"},{"closepx":"11.44","date":"2024/12/05","extMainIn":"-1286.86万","largeNetIn":"+4809.78万","littleNetIn":"+1387.22万","mediumNetIn":"-100.36万","ratio":"-0.17%","showtime":"2024-12-05","superNetIn":"-6096.64万","time":"20241205"},{"closepx":"11.46","date":"2024/12/04","extMainIn":"-4527.96万","largeNetIn":"+2466.81万","littleNetIn":"+4155.44万","mediumNetIn":"+372.53万","ratio":"-0.26%","showtime":"2024-12-04","superNetIn":"-6994.77万","time":"20241204"}],"info":{"code":"000001","name":"平安银行","price":"11.38","range":"-0.05","ratio":"-0.44%","stockstatus":"ENDTR","timestamp":"--"},"tab":"day","tabs":[{"text":"日资金流向","type":"day"},{"text":"周资金流向","type":"week"},{"text":"月资金流向","type":"month"}],"titleList":[{"isSort":"0","needColor":"0","sortKey":"date","text":"日期"},{"isSort":"0","needColor":"1","sortKey":"closepx","text":"股价"},{"isSort":"0","needColor":"1","sortKey":"ratio","text":"涨跌幅"},{"isSort":"0","needColor":"1","sortKey":"extMainIn","text":"资金净流入"},{"isSort":"0","needColor":"1","sortKey":"superNetIn","text":"特大单净流入"},{"isSort":"0","needColor":"1","sortKey":"largeNetIn","text":"大单净流入"},{"isSort":"0","needColor":"1","sortKey":"mediumNetIn","text":"中单净流入"},{"isSort":"0","needColor":"1","sortKey":"littleNetIn","text":"小单净流入"}]}}
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "content-type":"application/json; charset=utf-8",
    "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Cookie":"ZD_ENTRY=google; __bid_n=1920040598e8432c03408e; BIDUPSID=D29679C25841B3270013CB030662B3BF; PSTM=1732696944; BDRCVFR[S_ukKV6dOkf]=mk3SLVN4HKm; H_PS_PSSID=60279_61027_61141_61162_61203_61206_61213_61212_61208_61244_61190_61286; delPer=0; PSINO=3; BAIDUID=D29679C25841B3270013CB030662B3BF:SL=0:NR=10:FG=1; BAIDUID_BFESS=D29679C25841B3270013CB030662B3BF:SL=0:NR=10:FG=1; ZFY=zvL5ky9U4QC:Ag6wXDxiFDOhuQpm1TWjST54nJ8HhalA:C",
}

data ={'code': '300059'}
response = requests.get(url, headers=headers,data=data)
result = response.json()

print(response.status_code)
#user-agent:
#

