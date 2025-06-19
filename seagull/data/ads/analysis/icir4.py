# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 21:42:55 2025

@author: awei
icir4
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os
import multiprocessing


class evaluation:

    def __init__(self, df: pd.DataFrame, factor: str) -> None:
        """
        data:pd.DataFrame
        factor: factor name
        """
        self.data = df
        self.factors = factor
        self.ICstat = pd.DataFrame()
        self.REGstat = pd.DataFrame()
        self.GROUPstat = pd.DataFrame()
        self.RETstat = pd.DataFrame()

    @classmethod
    def _func_icir(cls, x, name):
        return x[name].corr(x['next_ret'])

    @classmethod
    def _regression_rlm(cls, X, name):
        X[name] = (X[name] - X[name].mean()) / X[name].std()
        x = X[name].values
        y = X['next_ret'].values
        x = sm.add_constant(x)
        model = sm.RLM(y, x, M=sm.robust.norms.HuberT()).fit()
        return model.params[-1], model.tvalues[-1]

    @classmethod
    def _func_group(cls, x, name, num):
        x['group'] = pd.qcut(x[name].rank(method='first'), q=num, labels=False)
        ret_list = []
        for i in range(num):
            ret = x.loc[x['group'] == i, 'next_ret'].mean()
            ret_list.append(ret)
        return ret_list

    @classmethod
    def _cal_group(cls, x):
        df = pd.DataFrame(x)
        num = len(df.iloc[0, 0])
        ret_df = pd.DataFrame(index=df.index)
        for i in range(num):
            ret_df[f"group_{i+1}"] = df[0].apply(lambda x: x[i])
        final_ret = (ret_df + 1).cumprod()
        final_ret = final_ret.iloc[-1].tolist()
        if final_ret[0] < final_ret[-1]:
            label = '正'
            ret_df[
                'group_longshort'] = ret_df[f'group_{num}'] - ret_df["group_1"]
        else:
            label = '反'
            ret_df[
                'group_longshort'] = ret_df["group_1"] - ret_df[f'group_{num}']

        return ret_df, label

    def calculate_ICIR(self):
        df = self.data.copy()
        ic_data = df.groupby('交易日期').apply(self._func_icir, self.factors)
        mean_ic = np.mean(ic_data)
        if mean_ic > 0:
            ic_pr = ic_data[ic_data > 0].shape[0] / ic_data.shape[0]
        else:
            ic_pr = ic_data[ic_data < 0].shape[0] / ic_data.shape[0]
        ir = abs(mean_ic) / ic_data.std()

        self.ICstat.loc[self.factors, '因子IC均值'] = mean_ic
        self.ICstat.loc[self.factors, 'IC_概率(均值>0或小于0)'] = ic_pr
        if abs(mean_ic) > 0.02:
            self.ICstat.loc[self.factors,
                            '因子IC评价(大于0.02(可用)大于0.04(较好))'] = '可用'
        elif abs(mean_ic) > 0.04:
            self.ICstat.loc[self.factors,
                            '因子IC评价(大于0.02(可用)大于0.04(较好))'] = '较好'
        else:
            self.ICstat.loc[self.factors,
                            '因子IC评价(大于0.02(可用)大于0.04(较好))'] = '不好'
        self.ICstat.loc[self.factors, '因子IC绝对值>0.02的概率'] = ic_data[
            abs(ic_data) > 0.02].shape[0] / ic_data.shape[0]
        self.ICstat.loc[self.factors, '因子IC绝对值>0.04的概率'] = ic_data[
            abs(ic_data) > 0.04].shape[0] / ic_data.shape[0]
        self.ICstat.loc[self.factors, '因子IR'] = ir
        self.ICstat = self.ICstat

    def regression_method(self):
        df = self.data.copy()
        factor_k = df.groupby('交易日期').apply(self._regression_rlm, self.factors)
        factor_k = pd.DataFrame(factor_k)
        factor_k['k'] = factor_k[0].apply(lambda x: x[0])
        factor_k['t'] = factor_k[0].apply(lambda x: x[1])
        del factor_k[0]
        ret_mean = factor_k['k'].mean()
        ret_t = np.abs(ret_mean) / factor_k['k'].std()
        factor_t_abs = factor_k['t'].abs().mean()
        self.REGstat.loc[self.factors, '因子平均收益'] = ret_mean
        self.REGstat.loc[self.factors, '因子收益t值'] = ret_t
        self.REGstat.loc[self.factors, 't值绝对值'] = factor_t_abs
        if ret_mean > 0:
            self.REGstat.loc[self.factors, '因子均值大于或小于0的概率'] = factor_k[
                factor_k['k'] > 0].shape[0] / factor_k.shape[0]
            self.REGstat.loc[self.factors, '因子均值大于或小于0t值大于2的概率'] = factor_k[
                factor_k['t'] > 2].shape[0] / factor_k.shape[0]
        else:
            self.REGstat.loc[self.factors, '因子均值大于或小于0的概率'] = factor_k[
                factor_k['k'] < 0].shape[0] / factor_k.shape[0]
            self.REGstat.loc[self.factors, '因子均值大于或小于0t值大于2的概率'] = factor_k[
                factor_k['t'] < -2].shape[0] / factor_k.shape[0]
        self.REGstat = self.REGstat

    def grouping(self, ngroup: int):
        df = self.data.copy()
        group_ret = df.groupby('交易日期').apply(self._func_group, self.factors,
                                             ngroup)
        group_ret, labal = self._cal_group(group_ret)
        self.GROUPstat.loc[self.factors, '因子方向'] = labal
        self.GROUPstat.loc[self.factors,
                           '多空收益'] = (group_ret['group_longshort'] +
                                      1).prod() - 1
        self.GROUPstat.loc[self.factors, '多空最大回撤'] = (
            (group_ret['group_longshort'] + 1).cumprod() /
            (group_ret['group_longshort'] + 1).cumprod().expanding().max() -
            1).min()
        group_ret.index = pd.to_datetime(group_ret.index)
        self.GROUPstat.loc[self.factors, '多空夏普'] = (
            group_ret['group_longshort'] + 1).cumprod().diff().mean() / (
                group_ret['group_longshort'] +
                1).cumprod().diff().std() * np.sqrt(
                    365 / (group_ret.index.tolist()[1] -
                           group_ret.index.tolist()[0]).days)

        self.RETstat = group_ret

    def draw_picture(self, if_save=True):
        df = self.RETstat.copy()
        fig1 = plt.figure(figsize=(18, 9))
        ax1 = plt.subplot(2, 1, 1)
        ax1.set_title("group_equity")
        for col in df.columns:
            plt.plot((df[col] + 1).cumprod(), label=col)
        plt.legend(loc='best')

        ax2 = plt.subplot(2, 1, 2)
        ax2.set_title("group_net")
        x = df.columns.tolist()
        y = np.array((df + 1).prod().tolist())
        plt.bar(x, y)
        plt.legend(loc='best')
        if if_save:
            if not os.path.exists(f"{os.getcwd()}/data/output/{self.factors}"):
                os.mkdir(f"{os.getcwd()}/data/output/{self.factors}")

            plt.savefig(
                f"{os.getcwd()}/data/output/{self.factors}/{self.factors}.jpg")
            res = pd.concat([self.ICstat, self.REGstat, self.GROUPstat],
                            axis=1)
            res.to_csv(
                f"{os.getcwd()}/data/output/{self.factors}/{self.factors}.csv",
                index=0)

        plt.show()


def run_full_func(factor, if_pro=True):
    if if_pro:
        process1 = multiprocessing.Process(target=factor.calculate_ICIR())
        process2 = multiprocessing.Process(target=factor.regression_method())
        process3 = multiprocessing.Process(target=factor.grouping(5))

        process1.start()
        process2.start()
        process3.start()

        process1.join()
        process1.join()
        process1.join()

        factor.draw_picture()
    else:
        factor.calculate_ICIR()
        factor.regression_method()
        factor.grouping(5)
        factor.draw_picture()

import pandas as pd
import numpy as np
import itertools, os, time
from factorevaluation.one_factor import *


class multifactor:

    def __init__(self, data: pd.DataFrame, factor_list: list) -> None:
        self.data = data
        self.factor_list = factor_list

    @classmethod
    def _calculate_corr(cls, x: pd.DataFrame, combin):
        x.sort_values(['交易日期'], inplace=True)
        com_dict = {}
        for com in combin:
            corr = x[com[0]].corr(x[com[1]])
            com_dict[com] = corr
        return com_dict

    @classmethod
    def _find_one_factor_corr(cls, name: tuple, corr: dict):
        res = {}
        for n in name:
            corr_list = []
            for k, v in corr.items():
                if (n in k) and (name != k):
                    corr_list.append(v)
            res[n] = np.mean(corr_list)
        if res[name[0]] > res[name[1]]:
            return name[0]
        else:
            return name[1]

    @classmethod
    def _delete_corr(cls, name, corr):
        new_corr = {}
        for k, v in corr.items():
            if name not in k:
                new_corr[k] = v
        return new_corr

    @classmethod
    def _factor_label(cls, name):
        if not os.path.exists(
                f"/home/tradingking/python/factorevaluation/data/output/{name}"
        ):
            print(f"{name}文件不存在")
            exit()
        df = pd.read_csv(
            f"/home/tradingking/python/factorevaluation/data/output/{name}/{name}.csv"
        )
        label = df['因子方向'].iat[0]
        return label

    @classmethod
    def _calculate_new_factor(cls, df: pd.DataFrame, label: dict):
        for k, v in label.items():
            if v == '正':
                df[f"{k}_z"] = (df[k] - df[k].mean()) / df[k].std()
            else:
                df[f"{k}_z"] = -((df[k] - df[k].mean()) / df[k].std())
        new_list = [name + "_z" for name in label.keys()]

        df[f"{'_'.join(label.keys())}"] = df[new_list].sum(axis=1)
        return df[['交易日期', '股票代码', 'next_ret'] + [f"{'_'.join(label.keys())}"]]

    # 1.等权
    def equal_weight_factor(self):
        df = self.data.copy()
        combin = list(itertools.combinations(self.factor_list, 2))
        corr_df = df.groupby('股票代码').apply(self._calculate_corr, combin)
        corr_dict = {}
        for com in combin:
            corr = corr_df.apply(lambda x: x[com])
            corr_dict[com] = corr.mean()
        del_list = []
        for k, v in corr_dict.items():
            for j in k:
                if j in del_list:
                    continue
            if v >= 0.5:
                del_name = self._find_one_factor_corr(k, corr_dict)
                corr_dict = self._delete_corr(del_name, corr_dict)
                del_list.append(del_name)
        new_factor = []
        for name in self.factor_list:
            if name not in del_list:
                new_factor.append(name)
        print(f"因子组合列表:{new_factor}")
        factor_label = {}
        for name in new_factor:
            factor_label[name] = self._factor_label(name)
        df = df[['交易日期', '股票代码', 'next_ret'] + new_factor]
        new_df = df.groupby('交易日期').apply(self._calculate_new_factor,
                                          factor_label)
        new_name = new_df.columns.tolist()[-1]
        one = evaluation(new_df, new_name)
        run_full_func(one)

if __name__ == "__main__":
    import pandas as pd
    import time
    df = pd.read_pickle("data/all_stock_data_W.pkl")
    # df = df[df['交易日期'] >= pd.to_datetime("20160101")]
    df['next_ret'] = df['下周期每天涨跌幅'].apply(lambda x: np.prod(np.array(x) + 1) - 1)
    del df['下周期每天涨跌幅']
    df.dropna(inplace=True)
    factors = ['总市值', '换手率mean_20', '量价相关系数_5', '流通市值', '成交额std_5', '成交额std_20']
    df = df[['交易日期', '股票代码', 'next_ret'] + factors]
    s = time.time()
    factor = multifactor(df, factor_list=factors)
    factor.equal_weight_factor()
    print(f"合计用时 - {time.time()-s} s")