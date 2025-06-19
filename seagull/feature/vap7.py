# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 21:46:27 2024

@author: awei
vap7
"""
import os
import copy
import argparse

import pandas as pd
import numpy as np
import copy

from __init__ import path
from utils import utils_database, utils_log

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{path}/log/{log_filename}.log')


class ChipDistribution():

    def __init__(self):
        self.Chip = {} # 当前获利盘
        self.ChipList = {}  # 所有的获利盘的

   # def get_data(self):
   #     self.



    # Triangular distribution
    def average_distribution(self, date_1, high_1, low_1, volume_1, turnover_1, k, min_price_unit):
        # 平均分布,Average distribution
        # 生成价格区间的价格点
        price_points = np.arange(low_1, high_1, min_price_unit)
        add_chips = volume_1 / len(price_points)
        print('price_points', len(price_points))
        print('add_chips', add_chips)
        # 将当前的 self.Chip 转换为 DataFrame
        if not hasattr(self, 'Chip_df'):
            # 初始化 DataFrame，将 self.Chip 转换为 DataFrame 格式
            self.Chip_df = pd.DataFrame(list(self.Chip.items()), columns=['price', 'chip'])
        else:
            # 如果 Chip_df 已存在，先更新稀释
            self.Chip_df['chip'] *= (1 - turnover_1 * k)
    
        # 创建新增的筹码 DataFrame
        new_chips_df = pd.DataFrame({
            'price': price_points,
            'chip': add_chips * (turnover_1 * k)
        })
    
        # 使用 merge 操作更新 self.Chip_df，将新增筹码合并进原有的筹码分布
        self.Chip_df = pd.merge(self.Chip_df, new_chips_df, on='price', how='outer', suffixes=('', '_new'))
        self.Chip_df['chip'].fillna(0, inplace=True)
        self.Chip_df['chip_new'].fillna(0, inplace=True)
        self.Chip_df['chip'] += self.Chip_df['chip_new']
        self.Chip_df.drop(columns='chip_new', inplace=True)
        
        # 将更新后的筹码分布转回字典并保存
        self.Chip = dict(zip(self.Chip_df['price'], self.Chip_df['chip']))
        self.ChipList[date_1] = copy.deepcopy(self.Chip)

# =============================================================================
#     def average_distribution(self,date_1,high_1, low_1, volume_1, turnover_1, k, min_price_unit):
# 
#         x =[]
#         intervals_num = (high_1 - low_1) / min_price_unit
#         for i in range(int(intervals_num)):
#             x.append(round(low_1 + i * min_price_unit, 2))
#         add_chips = volume_1/len(x)
#         for i in self.Chip:
#             self.Chip[i] = self.Chip[i] *(1 -turnover_1 * k)
#         for i in x:
#             if i in self.Chip:
#                 self.Chip[i] += add_chips *(turnover_1 * k)
#             else:
#                 self.Chip[i] = add_chips *(turnover_1 * k)
#         
#         self.ChipList[date_1] = copy.deepcopy(self.Chip)
# =============================================================================


    def calcuChip(self,data, flag=1, AC=1):  #flag 使用哪个计算方式,    AC 衰减系数
        low = self.data['low']
        high = self.data['high']
        volume = self.data['volume']
        turnover = self.data['turnover']
        avg = self.data['avg_price']
        date = self.data['date']

        for i in range(len(date)):
        #     if i < 90:
        #         continue

            high_1 = high[i]
            low_1 = low[i]
            volume_1 = volume[i]
            turnover_1 = turnover[i]
            #avg_price_1 = avg_price[i]
            # print(date[i])
            date_1 = date[i]
            self.average_distribution(date_1,high_1, low_1, volume_1, turnover_1, k=1, min_price_unit=0.01)
            
        # 计算winner
    def winner(self,p=None):
            Profit = []
            date = self.data['date']

            if p == None:  # 不输入默认close
                p = self.data['close']
                count = 0
                for i in self.ChipList:
                    # 计算目前的比例

                    Chip = self.ChipList[i]
                    total = 0
                    be = 0
                    for i in Chip:
                        total += Chip[i]
                        if i < p[count]:
                            be += Chip[i]
                    if total != 0:
                        bili = be / total
                    else:
                        bili = 0
                    count += 1
                    Profit.append(bili)
            else:
                for i in self.ChipList:
                    # 计算目前的比例

                    Chip = self.ChipList[i]
                    total = 0
                    be = 0
                    for i in Chip:
                        total += Chip[i]
                        if i < p:
                            be += Chip[i]
                    if total != 0:
                        bili = be / total
                    else:
                        bili = 0
                    Profit.append(bili)

            # import matplotlib.pyplot as plt
            # plt.plot(date[len(date) - 200:-1], Profit[len(date) - 200:-1])
            # plt.show()

            return Profit

    def lwinner(self,N = 5, p=None):

        data = copy.deepcopy(self.data)
        date = data['date']
        ans = []
        for i in range(len(date)):
            print(date[i])
            if i < N:
                ans.append(None)
                continue
            self.data = data[i-N:i]
            self.data.index= range(0,N)
            self.__init__()
            self.calcuChip()    #使用默认计算方式
            a = self.winner(p)
            ans.append(a[-1])
        import matplotlib.pyplot as plt
        plt.plot(date[len(date) - 60:-1], ans[len(date) - 60:-1])
        plt.show()

        self.data = data
        return ans



    def cost(self,N):
        date = self.data['date']

        N = N / 100  # 转换成百分比
        ans = []
        for i in self.ChipList:  # 我的ChipList本身就是有顺序的
            Chip = self.ChipList[i]
            ChipKey = sorted(Chip.keys())  # 排序
            total = 0  # 当前比例
            sumOf = 0  # 所有筹码的总和
            for j in Chip:
                sumOf += Chip[j]

            for j in ChipKey:
                tmp = Chip[j]
                tmp = tmp / sumOf
                total += tmp
                if total > N:
                    ans.append(j)
                    break
        import matplotlib.pyplot as plt
        plt.plot(date[len(date) - 1000:-1], ans[len(date) - 1000:-1])
        plt.show()
        return ans



if __name__ == "__main__":
    data = pd.read_csv(f'{path}/data/test.csv')
    data.turnover = data.turnover/100
    low_1,high_1,volume_1,turnover_1,avg_price_1,date_1 = data.loc[0,['low','high','volume','turnover','avg_price','date']]

    chip_distribution = a=ChipDistribution()
    chip_distribution.average_distribution(date_1=date_1,
                                           high_1=high_1,
                                           low_1=low_1,
                                           volume_1=volume_1,
                                           turnover_1=turnover_1,
                                           k=1,
                                           min_price_unit=0.01,
                                           )

    #a.get_data() #获取数据
    #a.calcuChip(data, flag=1, AC=1) #计算
    #a.winner() #获利盘
    #a.cost(90) #成本分布



    #a.lwinner()
