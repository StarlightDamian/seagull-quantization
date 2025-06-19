# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 21:51:34 2024

@author: awei
vap8
"""

import  pandas as pd
path='D:/pycharmproject/筹码峰/'
def GetCirculatingStock(path,data):
    ## 获取每天对应的十大流通和流通股总数
    df=pd.read_csv(path+data+'.txt',encoding='gbk',usecols=['日期', '时间', '十大流通', '流通股总数','成交数量','成交金额'])
    CirculatingStock=df[['日期','十大流通', '流通股总数']].drop_duplicates().sort_values(by=['日期'])

    return CirculatingStock

def Compress(data):
    # 坐标（价格）压缩
    max_price = max(data['price'])
    min_price = min(data['price'])
    dis = round((max_price - min_price) / (100 - 1), 2)
    # 创建一个列表存放标准的刻度
    scale = []
    for i in range(100):
        s = min_price + i * dis
        scale.append(s)
    # 创建一个新的数组存储标准化后的筹码分布
    data_new = pd.DataFrame(columns=['date', 'price', 'num'])
    data_new['price'] = scale
    data_new['num'] = 0
    data_new['date'] = data['date'][0]
    # 遍历筹码分布，将每一个价格的筹码压缩到标准价格对应的筹码上
    for i in range(len(data)):
        p = data['price'][i]  # 第i个价格
        r = data['num'][i]  # 第i个价格对应的筹码数量
        for j in range(100):
            if p >= data_new['price'][j] and p < data_new['price'][j + 1]:
                data_new['num'][j] = data_new['num'][j] + round((1 - abs(data_new['price'][j] - p) / dis) * r,0)
                data_new['num'][j + 1] = data_new['num'][j + 1] + round((1 - abs(p - data_new['price'][j + 1]) / dis) * r,0)
                break
            else:
                continue

    return data_new

def CalChipDistribution(path,data,初始值,a):
    #计算股票每天各个价格对应的筹码分布情况
    df=pd.read_csv(path+data+'.txt',encoding='gbk',usecols=['日期', '时间', '十大流通', '流通股总数','成交数量','成交金额'])
    initialValue=pd.read_csv(path+初始值+'.csv',encoding='gbk')

    df_day=df[['日期','成交数量','成交金额']]
    df_day_lt=df[['日期','十大流通', '流通股总数']].drop_duplicates()
    df_day=df_day.groupby('日期').sum()
    #计算日均价
    df_day['日均价']=df_day['成交金额']/(df_day['成交数量']*100)
    df_day=df_day.reset_index()
    df_day=df_day.merge(df_day_lt,how='left',on='日期')
    #计算每日换手率
    df_day['换手率']=(df_day['成交数量']*100)/(df_day['流通股总数']-df_day['十大流通'])

    #获取总共交易了多少天
    rows=df_day.shape[0]
    #创建一个dataframe存储筹码分布
    chips=pd.DataFrame(columns={'date':'','price':'','num':''})
    #将初始值加载到chips中
    for i in range (len(initialValue)):
        temp={'date':initialValue['日期'][i],'price':initialValue['价格'][i],'num':initialValue['数量'][i]}
        chips=chips.append(temp,ignore_index=True)

    #创建一个dataframe存储所有的筹码分布
    chips_all=pd.DataFrame(columns={'date':'','price':'','num':''})
    for i in range(0,rows):
        price=df_day['日均价'][0]
        if price not in chips['price'].tolist():        #当价格不在已有的筹码分布里面
            temp={'date':df_day['日期'][0],'price':price,'num':0}
            chips=chips.append(temp,ignore_index=True) #把这个价格添加到筹码分布中
            chips=chips.sort_values(by=['price'])
            #其他价格的筹码数量
            chips['num']=chips['num']*(1-df_day['换手率'][i]*a)
            priceindex=chips.loc[chips['price']==price].index
            #新价格的筹码数量
            chips['num'][priceindex]=df_day['换手率'][i]*a*(df_day['流通股总数'][i]-df_day['十大流通'][i])
            #日期修改为当前日期
            chips['date']=df_day['日期'][i]
        else:   #价格在已有的筹码分布里面
            chips['num'] = chips['num'] * (1 - df_day['换手率'][i] * a)
            priceindex = chips.loc[chips['price'] == price].index
            #当日价格的筹码数量
            chips['num'][priceindex]=df_day['换手率'][i]*a*(df_day['流通股总数'][i]-df_day['十大流通'][i])+chips['num'][priceindex]
            #日期修改为当前日期
            chips['date']=df_day['日期'][i]
            chips = chips.sort_values(by=['price'])
        #判断需不需要进行价格压缩
        if len(chips)<=100:
            chips_all = pd.concat([chips_all, chips], ignore_index=True)
        else:
            chips=Compress(chips)   # 进行坐标价格压缩
            chips_all = pd.concat([chips_all, chips], ignore_index=True)
    result['date'] = [str(date) for date in result['date']]
    return chips_all


result=CalChipDistribution(path,'data','初始值',1.1)
GetCirculatingStock(path,'data')
Compress(chips)