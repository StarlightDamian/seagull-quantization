# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 10:20:22 2025

@author: awei
demo_alpha101_arrayd
"""

import pandas as pd
import numpy as np
from seagull.settings import PATH

import os
import argparse
from datetime import datetime

import pandas as pd
import numpy as np
import numpy.lib.stride_tricks as sk
from scipy.stats import rankdata
import efinance as ef  # efinance不能连国际VPN

from seagull.settings import PATH
from seagull.utils import utils_database, utils_log, utils_data

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')


def ts_sum(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    滚动求和：计算过去 window 个交易日的累积和。
    :param df: pandas DataFrame，行索引为日期，列索引为股票/品种等。
    :param window: 滚动窗口大小（天数）。
    :return: 每个日期的“过去 window 天”窗口内求和结果。
    """

    return df.rolling(window).sum()


def sma(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    简单移动平均：计算简单移动平均（SMA）。
    :param df: pandas DataFrame，行情数据等。
    :param window: 滚动窗口大小。
    :return: 每个日期的“过去 window 天”平均值。
    """
    return df.rolling(window).mean()


def stddev(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    滚动标准差：计算滚动标准差（波动率）。
    :param df: a pandas DataFrame.
    :param window: 滚动窗口大小。
    :return: 每个日期过去 window 天的标准差。
    """
    return df.rolling(window).std()


def ts_min(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    滚动最小值：计算过去 window 天的最小值。
    :param df: a pandas DataFrame.
    :param window: 滚动窗口大小。
    :return: 每个日期过去 window 天的最小值。
    """
    return df.rolling(window).min()


def ts_max(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    滚动最大值：计算过去 window 天的最大值。
    :param df: a pandas DataFrame.
    :param window: 滚动窗口大小。
    :return: 每个日期过去 window 天的最大值。
    """
    return df.rolling(window).max()


def correlation(x: pd.DataFrame, y: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    滚动相关系数：计算 x 与 y 在过去 window 天的逐日滚动皮尔逊相关系数。
    :param x: pandas DataFrame 或 Series，行情/因子序列。
    :param y: pandas DataFrame 或 Series，与 x 形状需对应（同索引）。
    :param window: 滚动窗口大小。
    :return:  DataFrame 或 Series，表示每个日期的滚动相关系数。
    """
    return x.rolling(window).corr(y)


def covariance(x: pd.DataFrame, y: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    滚动协方差：计算 x 与 y 在过去 window 天的滚动协方差。
    :param x: pandas DataFrame 或 Series。
    :param y: pandas DataFrame 或 Series，与 x 索引对齐。
    :param window: 滚动窗口大小。
    :return: 每个日期的滚动协方差。
    """
    return x.rolling(window).cov(y)


def ts_rank(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    滚动排序/分位：计算每个日期对应值在过去 window 天内的排名百分比。
    :param df: pandas DataFrame，行索引为日期。
    :param window: 滚动窗口大小。
    :return: DataFrame，与 df 同形状，每个元素表示该列在过去 window 天的 ts_rank。
    """
    def rolling_rank(arr: np.ndarray) -> float:
        """
        帮助函数：返回 arr（长度为 window）中最后一个元素的横截面排序位置（百分比）。
        :param arr: numpy 数组，长度等于窗口大小。
        :return: float，表示最后一个元素在 arr 中的 rank pct。
        """
        # rankdata 默认给出 1 到 n 的排名，这里以百分比形式返回
        ranks = rankdata(arr)  # 生成 [1, 2, ..., window] 排名
        return ranks[-1] / len(arr)  # 将 rank 转换为 [1/window, 2/window, ...]，最后一个位置的值

    # 直接调用 rolling.apply 并传入自定义函数
    # 注意：rolling.apply 默认 axis=0，对每列做窗口，rolling_rank 会得到最后一个值的排名百分比
    return df.rolling(window).apply(rolling_rank, raw=True)


def product(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    滚动乘积：计算过去 window 天每列数据的滚动乘积。
    :param df: a pandas DataFrame.
    :param window: 窗口大小。
    :return: 每个日期的过去 window 天的乘积。
    """
    def rolling_prod(arr: np.ndarray) -> float:
        """
        返回数组 arr 中所有值的乘积。
        :param na: numpy 数组。
        :return: float，乘积结果。
        """
        return np.prod(arr)

    # raw=True：直接传入 numpy 数组给 rolling_prod，更省内存
    # 对于长度很长的窗口（比如 N>100），直接 np.prod 可能会导致数值溢出，建议先对数 sum 再 exp。
    # return np.exp(df.rolling(window).apply(lambda arr: np.log(arr).sum()))
    return df.rolling(window).apply(rolling_prod, raw=True)


def delta(df: pd.DataFrame, period: int = 1) -> pd.DataFrame:
    """
    差分：计算当前值与 period 天前的差：df_t - df_{t-period}。
    :param df: a pandas DataFrame.
    :param period: 滞后天数。
    :return: 每列对应的差分结果，首 period 行会产生 NaN。
    """
    return df.diff(period)


def delay(df, period=1):
    """
    滞后：将整个时序向下平移 period 行，用于计算滞后值。
    :param df: a pandas DataFrame.
    :param period: 滞后行数。
    :return: df.shift(period)，前 period 行为 NaN
    """
    return df.shift(period)


def rank(df):
    """
    横截面排序：计算每个日期横截面上，每列在当日所有列中的分位排名（百分比形式）。
    :param df: pandas DataFrame，行索引为日期，列为不同股票/品种。
    :return: DataFrame，与 df 同形状，每列为该值在当日所有列中的排名百分比。=
    :优化：如果仅想在某一组股票池内做排名，可先用 df[chosen_universe] 再做 rank。
    """
    # return df.rank(axis=1, pct=True)
    return df.rank(pct=True)


def scale(df, k=1):
    """
    归一化，使绝对值之和为 k。将每列按绝对值之和归一化，使得 sum(abs(df)) = k。常用于因子值归一化。
    :param df: a pandas DataFrame.
    :param k: 归一化后总绝对值之和（例如 k=1 表示 L1 归一化）。
    :return: DataFrame，与 df 同形状。
    优化：1.如果列中存在 NaN，建议先用 .fillna(0) 或相似操作处理。
         2.如果只想对行进行归一化（比如当日所有股票加起来为 1），应把 axis=1 作为分母。
    """
    # abs(df).sum()：按列计算绝对值之和
    # df.div(...)：逐列除以当列的绝对值之和
    return df.mul(k).div(np.abs(df).sum(axis=0))


def ts_argmax(df, window=10):
    """
    滚动最大值所在位置：计算每个日期过去 window 天内的最大值出现在距离当前有几天之前。
    :param df: a pandas DataFrame.
    :param window: 滚动窗口大小。
    :return: DataFrame，返回值范围为 1 到 window，1 表示当日最大，window 表示 window 天前最大。
    优化：1.raw=True：可提升传入 numpy 数组的效率。
         2.需要注意，rolling.apply 对于大数据量会比较慢，因为底层会对每个窗口调用 Python 回调。如果 N 很大或样本很大，需要用 numba 重写、或者拆解为分块计算。
    """
    # rolling.apply 会把窗口内数据 (长度=window) 传给 np.argmax
    # np.argmax 返回索引 0..window-1，需要 +1 让它变为 1..window
    # return df.rolling(window).apply(np.argmax) + 1
    return df.rolling(window).apply(lambda arr: np.argmax(arr) + 1, raw=True)


def ts_argmin(df, window=10):
    """
    滚动最小值所在位置：计算每个日期过去 window 天内的最小值出现在距离当前有几天之前。
    :param df: a pandas DataFrame.
    :param window: 滚动窗口大小。
    :return: DataFrame，范围 1..window，1 表示当日最小，window 表示 window 天前最小。
    """
    # return df.rolling(window).apply(np.argmin) + 1
    return df.rolling(window).apply(lambda arr: np.argmin(arr) + 1, raw=True)


def decay_ewm(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    用指数加权移动平均（EWMA）近似线性加权效果。并非线性权重，但常被用作替代
    :param df: pandas DataFrame
    :param window: EWM span
    :return: EWM 结果
    """
    return df.ewm(span=window, adjust=False).mean()


# def decay_linear(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
#     """
#     用 numpy sliding_window_view + 矩阵运算做线性加权平均，加速大规模计算。numpy 加速线性加权
#     :param df: pandas DataFrame
#     :param window: 窗口大小
#     :return: DataFrame
#     """
#     arr = df.to_numpy()  # shape (n_days, n_cols)
#     n_days, n_cols = arr.shape
#
#     # 首先创建一个 shape=(n_days-period+1, period, n_cols) 的滑窗视图
#     # sliding_window_view 需要 numpy >= 1.20
#     windows = sk.sliding_window_view(arr, window_shape=(window, n_cols))  # 结果维度需要按列分片
#     # 结果 shape = (n_days-period+1, period, n_cols)
#     windows = windows[:, :, :]  # 直接使用
#
#     # 计算线性权重（长度为 period）
#     weights = np.arange(1, period + 1, dtype=np.float64)
#     weights = weights / weights.sum()  # 归一化
#
#     # 对每个日期批量做加权求和
#     # windows 是 (有效天数, period, n_cols)，要对第二维 (period) 做加权
#     weighted = np.tensordot(windows, weights, axes=([1], [0]))
#     # weighted shape = (n_days-period+1, n_cols)
#
#     # 最终结果要拼回成 (n_days, n_cols)，前面的 period-1 行可置 NaN 或原始值
#     out = np.full((n_days, n_cols), np.nan)
#     out[window-1:] = weighted
#
#     return pd.DataFrame(out, index=df.index, columns=df.columns)


class AlphaCalculator:
    def __init__(self, stock_df, fill_method='ffill'):
        """
        stock_df: DataFrame，包含日期、股票ID、特征（如close, open, high, low）
        fill_method: 填补缺失值的方法，默认为'ffill'，表示前向填充。
        """
        self.stock_df = stock_df
        self.fill_method = fill_method
        self._prepare_data()
        self.close = self.stock_df['close']
        self.open = self.stock_df['open']
        self.high = self.stock_df['high']
        self.low = self.stock_df['low']
        self.volume = self.stock_df['volume']
        self.returns = self.stock_df['close_rate']
        self.vwap = self.stock_df['vwap']

    def _prepare_data(self):
        """
        将数据整理为一个二维DataFrame，按full_code和日期排列，填充缺失值
        """
        # 转换为多层索引，行是日期，列是full_code，列上是各个特征
        self.stock_df.set_index(['date', 'full_code'], inplace=True)
        
        # 确保数据按日期和股票ID排序
        self.stock_df.sort_index(inplace=True)
        
        # 将数据透视为（日期, 特征） -> (full_code)
        self.stock_df = self.stock_df.unstack(level='full_code')
        
        # 填充缺失值
        if self.fill_method == 'ffill':
            self.stock_df = self.stock_df.fillna(method='ffill', axis=0)
        elif self.fill_method == 'bfill':
            self.stock_df = self.stock_df.fillna(method='bfill', axis=0)
        else:
            self.stock_df = self.stock_df.fillna(0)

    def alpha101(self):
        """
        计算Alpha#101因子: ((close - open) / ((high - low) + 0.001))
        返回一个DataFrame，包含每个日期、每只的因子值
        """
        alpha_101 = (self.close - self.open) / ((self.high - self.low) + 0.001)
        return alpha_101
         
    
if __name__ == '__main__':
    # raw_df = pd.read_feather(f'{PATH}/_file/das_wide_incr_train_mini.feather')
    # # ['date', 'full_code', 'open', 'close', 'high', 'low', 'volume','turnover', 'prev_close', 'close_rate', 'vwap']
    # # raw_df[['prev_close']] = raw_df[['close']].shift(1)
    # # raw_df[['close_rate']] = raw_df[['close']].div(raw_df['prev_close'], axis=0)
    # # raw_df['vwap'] = raw_df['close']
    #
    #
    # # 计算Alpha#101
    # alpha_calculator = AlphaCalculator(raw_df)
    # alpha_101_result = alpha_calculator.alpha101()
    #
    # # 查看结果
    # print("Alpha101因子结果 (date * full_code):\n", alpha_101_result)

    with utils_database.engine_conn("POSTGRES") as conn:
        df = pd.read_sql("select * from ods_ohlc_incr_efinance_stock_daily where 股票代码 in ('000001','002594')", con=conn.engine)

    df = df.rename(columns={'股票名称': 'code_name',
                            '股票代码': 'asset_code',
                            '日期': 'date',
                            '开盘': 'open',
                            '收盘': 'close',
                            '最高': 'high',
                            '最低': 'low',
                            '成交量': 'volume',
                            '成交额': 'amount',
                            '振幅': 'amplitude',  # new
                            '涨跌幅': 'chg_rel',
                            '涨跌额': 'price_chg',  # new
                            '换手率': 'turnover',
                            })
    df = df[['date', 'asset_code', 'open', 'close', 'high', 'low', 'volume', 'turnover', 'amount']]
    df = df.rename(columns={'asset_code': 'full_code'})
    df['amount'] = df['amount'].astype('int64')
    df.to_feather(f'{PATH}/_file/das_wide_incr_train_mini.feather')



# # 示例数据
# data_dict = {
#     'date': pd.date_range(start='2021-01-01', periods=5, freq='D').tolist() * 5,
#     'full_code': ['A', 'B', 'C', 'D', 'E'] * 5,
#     'close': np.random.rand(25),
#     'open': np.random.rand(25),
#     'high': np.random.rand(25),
#     'low': np.random.rand(25),
# }
# 
# # 创建DataFrame
# stock_df = pd.DataFrame(data_dict)

# 我希望生成一个np.ndarray的alpha101_array，包含每个日期和full_code的alpha101因子值
# alpha101_array = alpha_calculator.alpha101().to_numpy()


# raw_df
# Out[4]:
#                         open   close    high  ...  prev_close  close_rate    vwap
# date       full_code                          ...
# 1991-04-03 000001      -2.22   -2.22   -2.22  ...      364.46   -0.006091   -2.22
# 1991-04-04 000001      -2.22   -2.22   -2.22  ...        1.66   -1.337349   -2.22
# 1991-04-05 000001      -2.22   -2.22   -2.22  ...       -2.22    1.000000   -2.22
# 1991-04-06 000001      -2.22   -2.22   -2.22  ...       -2.22    1.000000   -2.22
# 1991-04-08 000001      -2.22   -2.22   -2.22  ...       -2.22    1.000000   -2.22
#                       ...     ...     ...  ...         ...         ...     ...
# 2025-05-26 002594     401.81  381.00  403.00  ...      405.00    0.940741  381.00
# 2025-05-27 000001      11.45   11.49   11.54  ...       11.42    1.006130   11.49
#            002594     375.88  372.41  381.00  ...      381.00    0.977454  372.41
# 2025-05-28 000001      11.50   11.53   11.55  ...       11.49    1.003481   11.53
#            002594     372.41  364.46  372.95  ...      372.41    0.978653  364.46
# [11527 rows x 9 columns]
# 这个二维的df，我希望转化为三维的np.ndarray,一个维度是日期，一个维度是full_code，一个维度是特征（open、close、high、low、vwap等）
#
#
#
# import numpy as np
# import pandas as pd
#
# # 假设 raw_df 就是你给出的那个 DataFrame，索引为 (date, full_code)，列为各个特征
# # 比如：
# # raw_df.index.names == ["date", "full_code"]
# # raw_df.columns == ["open", "close", "high", "low", "volume", "vwap", ...]
#
# # 1. 明确“日期”和“full_code”的顺序
# #    取出所有唯一的日期和 full_code，并排序（顺序可以按需求自定，比如从小到大）
# dates = raw_df.index.get_level_values("date").unique().sort_values()
# codes = raw_df.index.get_level_values("full_code").unique().sort_values()
#
# # 2. 确定要保留的特征列表（假设就是 raw_df.columns）
# features = list(raw_df.columns)
#
# # 3. 准备一个空的 ndarray，形状为 (n_dates, n_codes, n_features)
# n_dates = len(dates)
# n_codes = len(codes)
# n_features = len(features)
#
# arr = np.full((n_dates, n_codes, n_features), np.nan, dtype=float)
#
# # 4. 把原始 DataFrame 重新索引到 全量日期 × 全量代码
# #    这样可以确保在某些日期或某些股票缺失时，对应位置为 NaN
# #    先把 raw_df 按照 index.levels 进行重建索引
# reindexed = raw_df.reindex(
#     pd.MultiIndex.from_product([dates, codes], names=["date", "full_code"])
# )
#
# # 5. 依次把每个特征 “unstack” 到 (date × full_code) 的矩阵，再填到对应的 numpy 维度里
# #    unstack 后，DataFrame 的形状是 (n_dates, n_codes)
# for i, feat in enumerate(features):
#     # 5.1 取得这一列的 Series，并转成 (date × full_code) 的 DataFrame
#     mat = reindexed[feat].unstack(level="full_code")  # shape = (n_dates, n_codes)
#     # 5.2 mat 的行索引就是 dates，列索引就是 codes（自动对齐了）
#     #     我们只需要把 mat.values 填到 arr[:,:,i] 即可
#     arr[:, :, i] = mat.values
#
# # 6. 此时，arr[d_idx, c_idx, f_idx] 就对应：
# #    - d_idx: dates[d_idx]
# #    - c_idx: codes[c_idx]
# #    - f_idx: features[f_idx]
# #
# #    比如要看第 10 天（dates[10]）第 3 只股票（codes[3]）的 close 值：
# #      close_index = features.index("close")
# #      value = arr[10, 3, close_index]
#
# # 7. 如果需要把 dates、codes、features 一并存下来，方便后续索引：
# #    例如：
# dates_array = np.array(dates)    # shape = (n_dates,)
# codes_array = np.array(codes)    # shape = (n_codes,)
# features_list = features         # length = n_features
#
# # 至此，你就得到了一个三维数组 arr，以及对应的坐标说明 (dates_array, codes_array, features_list)











