# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 18:35:26 2024

@author: awei

alpha191
"""
import sys
import numpy as np
import pandas as pd

from lib.base import FactorBase
from typing import Dict
from lib.ops.factor_ops import RANK, SMA, SUM, STD, CORR, COVARIANCE, DELTA, TS_MAX, TS_MIN, TS_RANK, MEAN, COUNT, \
    DECAYLINEAR, \
    MAX, DELAY, LOWDAY, HIGHDAY, IFELSE, REGBETA
# from factors.ops.factor_ops import _alpha191_143
from lib.utils.method_attrs import factor_attr

try:
    from qlib.data.ops import rolling_slope, rolling_rsquare, rolling_resi
    from qlib.data.ops import expanding_slope, expanding_rsquare, expanding_resi
except ImportError:
    print("部分因子依赖cython实现的滚动回归, 导入 qlib.data.ops.rolling_* 失败！")
    print("如果不想安装qlib, 请使用将 https://github.com/microsoft/qlib/tree/main/qlib/data/_libs 下的pyx文件编译")


# TODO: 1. DECAYLINEAR
# TODO: 2. CHECK NAN FILL ERROR LIKE pd.ewm
@factor_attr(max_depend=10, return_type="float")
def alpha191_001(data, corr_period=6):
    """
    (-1 * CORR( RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6))
    """
    data1 = np.log(data["volume"]).diff(periods=1).rank(axis=1, pct=True)
    data2 = ((data["close"] - data["open"]) / data["open"]).rank(axis=1, pct=True)
    alpha = - data1.rolling(corr_period).corr(data2)
    return alpha


@factor_attr(max_depend=10, return_type="float")
def alpha191_002(data):
    """
    (-1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1))
    """
    alpha = - (
            ((data["close"] - data["low"]) - (data["high"] - data["close"])) / (data["high"] - data["low"])
    ).diff(periods=1)
    return alpha


@factor_attr(max_depend=10, return_type="float")
def alpha191_003(data, sum_period=6):
    """
    SUM( (CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
    """
    delay1 = data["close"].shift(1)
    # CLOSE>DELAY(CLOSE,1)
    condition2 = data["close"] > delay1
    # CLOSE<DELAY(CLOSE,1)
    condition3 = data["close"] < delay1

    part1 = data["close"] - np.minimum(data["low"][condition2], delay1[condition2])
    part2 = data["close"] - np.maximum(data["high"][condition3], delay1[condition3])
    alpha = part1.fillna(0) + part2.fillna(0)
    alpha = alpha.rolling(sum_period).sum()
    return alpha


@factor_attr(max_depend=30, return_type="binary")
def alpha191_004(data, roll_close_long=8, roll_close_short=2, roll_vol=20):
    """
    ((((SUM(CLOSE, 8) / 8) + STD(CLOSE, 8)) < (SUM(CLOSE, 2) / 2))
    ? (-1 * 1) : (((SUM(CLOSE, 2) / 2) <
    ((SUM(CLOSE, 8) / 8) - STD(CLOSE, 8))) ? 1 : (((1 < (VOLUME / MEAN(VOLUME,20))) || ((VOLUME /
    MEAN(VOLUME,20)) == 1)) ? 1 : (-1 * 1))))
    -->
        if ((ROLLING_MEAN(8) + ROLLING_STD(8)) < ROLLING_MEAN(2)) {
            -1
        } else {
            if (ROLLING_MEAN(2) < (ROLLING_MEAN(8) - ROLLING_STD(8))) {
                1
            } else {
                if (1 <= VOL_TIMES(20)) {
                    1
                } else {
                    -1
                }
            }
        }
    -->
    fill with 1
    if ((ROLLING_MEAN(8) + ROLLING_STD(8)) < ROLLING_MEAN(2)) set to -1

    """
    roll_mean = data["close"].rolling(roll_close_long).mean()
    roll_std = data["close"].rolling(roll_close_long).std()
    roll_mean_short = data["close"].rolling(roll_close_short).mean()
    vol_times = data["volume"] / data["volume"].rolling(roll_vol).mean()
    na_map = vol_times.isna().astype(float).replace(1., np.nan)
    condition1 = (roll_mean + roll_std) < roll_mean_short
    condition2 = roll_mean_short < (roll_mean - roll_std)
    condition3 = 1 <= vol_times

    alpha = (~data["close"].isna()).astype(float).replace(0, np.nan)  # 全1, nan不变

    alpha[condition1] = -1
    alpha[~condition1 & ~condition2 & ~condition3] = -1
    return alpha + na_map


@factor_attr(max_depend=20, return_type="float")
def alpha191_005(data, roll_tscorr=5, roll_tsmax=3):
    """
    (-1 * TSMAX(
        CORR(TSRANK(VOLUME, 5), TSRANK(HIGH, 5), 5)
    , 3))
    """
    tv = data["volume"].rolling(roll_tscorr).rank(axis=0, pct=True)
    th = data["high"].rolling(roll_tscorr).rank(axis=0, pct=True)
    alpha = tv.rolling(roll_tscorr).corr(th)
    alpha = alpha.rolling(roll_tsmax).max()
    return alpha


@factor_attr(max_depend=10, return_type="float")
def alpha191_006(data, delta_period=4):
    """
    (RANK(SIGN(DELTA((((OPEN * 0.85) + (HIGH * 0.15))), 4)))* -1)
    -->
    RANK(
        SIGN(
            DELTA(OPEN * 0.85 + HIGH * 0.15, 4)
        )
    ) * -1
    """
    val = data["open"] * 0.85 + data["high"] * 0.15
    alpha = -1 * np.sign(val.diff(periods=delta_period)).rank(axis=1, pct=True)
    return alpha


@factor_attr(max_depend=10, return_type="float")
def alpha191_007(data, period=3):
    """
    ((RANK(MAX((VWAP - CLOSE), 3)) + RANK(MIN((VWAP - CLOSE), 3))) * RANK(DELTA(VOLUME, 3)))
    -->
    [RANK(MAX(VWAP - CLOSE, 3)) + RANK(MIN(VWAP - CLOSE, 3))] * RANK(DELTA(VOLUME, 3))
    """
    rkmax = (data["vwap"] - data["close"]).rolling(period).max().rank(axis=1, pct=True)
    rkmin = (data["vwap"] - data["close"]).rolling(period).min().rank(axis=1, pct=True)
    rkdelta = data["volume"].diff(periods=period).rank(axis=1, pct=True)
    alpha = (rkmax + rkmin) * rkdelta
    return alpha


@factor_attr(max_depend=10, return_type="float")
def alpha191_008(data, period=4):
    """
    RANK(DELTA((((HIGH + LOW) / 2) * 0.2) + (VWAP * 0.8), 4) * -1)
    -->
    RANK(
        DELTA([(HIGH + LOW) * 0.1 + VWAP * 0.8], 4) * -1
    )
    """
    alpha = -1 * (data["high"] + data["low"]) * 0.1 + data["vwap"] * 0.8
    alpha = alpha.diff(periods=period).rank(axis=1, pct=True)
    return alpha


@factor_attr(max_depend=10, return_type="float")
def alpha191_009(data, ewma_min=2, ewma_max=7):
    """
    SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,7,2)
    -->
    SMA(
	    ((HIGH + LOW) / 2 - (DELAY(HIGH, 1) + DELAY(LOW, 1)) / 2) * (HIGH - LOW) / VOLUME,
	7, 2)
    """
    part1 = (data["high"] + data["low"]) / 2 - (data["high"].shift(1) + data["low"].shift(1)) / 2
    part2 = (data["high"] - data["low"]) / data["volume"]
    alpha = (part1 * part2).ewm(alpha=ewma_min / ewma_max).mean()
    return alpha


@factor_attr(max_depend=30, return_type="float")
def alpha191_010(data, roll_retstd=20, roll_max=5):
    """
    (RANK(MAX(((RET < 0) ? STD(RET, 20) : CLOSE)^2),5))
    """
    ret = data["close"] / data["close"].shift() - 1

    condtion = (ret < 0)
    dtrue = ret.rolling(roll_retstd).std()
    alpha = dtrue.where(condtion, data["close"]) ** 2
    alpha = np.maximum(alpha, roll_max).rank(axis=1, pct=True)
    return alpha


@factor_attr(max_depend=10, return_type="float")
def alpha191_011(data, sum_period=6):
    """
    SUM(((CLOSE-LOW)-(HIGH-CLOSE))./(HIGH-LOW).*VOLUME,6)
    -->
    SUM(
	    (2 * CLOSE - LOW - HIGH) / (HIGH - LOW) * VOLUME
	, 6)
    """
    alpha = (2 * data["close"] - data["low"] - data["high"]) / (data["high"] - data["low"])
    alpha = alpha * data["volume"]
    alpha = alpha.rolling(sum_period).sum()
    return alpha


@factor_attr(max_depend=20, return_type="float")
def alpha191_012(data, mean_period=10):
    """
    (RANK((OPEN - (SUM(VWAP, 10) / 10)))) * (-1 * (RANK(ABS((CLOSE - VWAP)))))
    -->
    (
        RANK((OPEN - (SUM(VWAP, 10) / 10)))
    ) * (
        -1 * RANK(ABS(CLOSE - VWAP))
    )
    """
    alpha = data["open"] - data["vwap"].rolling(mean_period).mean()
    alpha = alpha.rank(axis=1, pct=True)
    alpha = alpha * (-1 * (data["close"] - data["vwap"]).abs().rank(axis=1, pct=True))
    return alpha


@factor_attr(max_depend=10, return_type="float")
def alpha191_013(data):
    """
    (HIGH * LOW)^0.5 - VWAP
    """
    alpha = (data["high"] * data["low"]) ** 0.5 - data["vwap"]
    return alpha


@factor_attr(max_depend=10, return_type="float")
def alpha191_014(data, delay_period=5):
    """
    CLOSE-DELAY(CLOSE,5)
    """
    alpha = data["close"] - data["close"].shift(delay_period)
    return alpha


@factor_attr(max_depend=10, return_type="float")
def alpha191_015(data):
    """
    OPEN/DELAY(CLOSE,1)-1
    """
    alpha = data["open"] / data["close"].shift(1) - 1
    return alpha


@factor_attr(max_depend=20, return_type="float")
def alpha191_016(data, coll_period=5):
    """
    -1 * TSMAX(
	    RANK(CORR(RANK(VOLUME), RANK(VWAP), 5))
	, 5)
    """
    alpha = data["volume"].rank(axis=1, pct=True).rolling(coll_period).corr(data["vwap"].rank(axis=1, pct=True))
    alpha = alpha.rank(axis=1, pct=True).rolling(coll_period).max()
    return -1 * alpha


@factor_attr(max_depend=20, return_type="float")
def alpha191_017(data, max_period=15, delta_period=5):
    """
    RANK(VWAP - MAX(VWAP, 15)) ^ DELTA(CLOSE, 5)
    """
    alpha = (data["vwap"] - data["vwap"].rolling(max_period).max()).rank(axis=1, pct=True)
    alpha = alpha ** data["close"].diff(periods=delta_period)
    return alpha


@factor_attr(max_depend=10, return_type="float")
def alpha191_018(data, delay_period=5):
    """
    CLOSE/DELAY(CLOSE,5)
    """
    alpha = data["close"] / data["close"].shift(delay_period)
    return alpha


@factor_attr(max_depend=10, return_type="float")
def alpha191_019(data, period: int = 6, use_vwap: bool = True):
    """
    if (CLOSE < DELAY(CLOSE, 5)) {
        (CLOSE - DELAY(CLOSE, 5)) / DELAY(CLOSE, 5)
    } else {
        if (CLOSE === DELAY(CLOSE, 5)) {
            0
        } else {
            (CLOSE - DELAY(CLOSE, 5)) / CLOSE
        }
    }
    """
    _p = "vwap" if use_vwap else "close"
    conditon = data[_p] < data[_p].shift(period)
    alpha = (data[_p] - data[_p].shift(period)) / data[_p].shift(period)
    part2 = (data[_p] - data[_p].shift(period)) / data[_p]
    alpha = alpha.where(conditon, part2)
    return alpha


@factor_attr(max_depend=10, return_type="float")
def alpha191_020(data, period: int = 6, use_vwap: bool = True):
    """
    (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100
    """
    _p = "vwap" if use_vwap else "close"
    alpha = (data[_p] - data[_p].shift(period)) / data[_p].shift(period) * 100
    return alpha


@factor_attr(max_depend=10, return_type="float")
def alpha191_021(data, period: int = 6, use_vwap: bool = True):
    """
    REGBETA(MEAN(CLOSE,6),SEQUENCE(6))
    """
    _p = "vwap" if use_vwap else "close"
    alpha = data[_p].rolling(period).mean()  # .rolling(period).apply(lambda x: np.polyfit(seq, x, 1)[0])
    _index = alpha.index.copy()
    alpha = alpha.apply(lambda x: pd.Series(rolling_slope(x.values, period)))
    alpha.index = _index
    return alpha


@factor_attr(max_depend=30, return_type="float")
def alpha191_022(data, period: int = 6, ewma_period=12, use_vwap: bool = True):
    """
    SMEAN(((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)-DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3)),12,1)
    -->
    SMEAN(
	    ((CLOSE - MEAN(CLOSE, 6)) / MEAN(CLOSE, 6) - DELAY((CLOSE - MEAN(CLOSE, 6)) / MEAN(CLOSE, 6), 3)),
	12, 1)
	SMA?
    """
    _p = "vwap" if use_vwap else "close"
    val_mean = data[_p].rolling(period).mean()
    alpha = (data[_p] - val_mean) / val_mean
    alpha = alpha - alpha.shift(3)
    alpha = alpha.ewm(alpha=1 / ewma_period).mean()
    return alpha


@factor_attr(max_depend=30, return_type="float")
def alpha191_023(data, ma_period=20, use_vwap: bool = True):
    """
    condition = CLOSE > DELAY(CLOSE, 1)
    std = STD(CLOSE, 20)
    SMA((condition ? std : 0), 20, 1)
        /
    (
        SMA((condition ? std : 0), 20, 1) +
        SMA((!condition ? std : 0), 20, 1)
    ) * 100
    @RES 0.004326   -0.000688
    """
    _p = "vwap" if use_vwap else "close"
    condition = data[_p] > data[_p].shift(1)
    std = data[_p].rolling(ma_period).std()
    part1 = std.where(condition, 0)
    part2 = std.where(~condition, 0)
    # alpha = part1.ewm(alpha=1 / ma_period).mean() / \
    #         (part1.ewm(alpha=1 / ma_period).mean() + part2.ewm(alpha=1 / ma_period).mean()) * 100
    alpha = SMA(part1, ma_period, 1) / (SMA(part1, ma_period, 1) + SMA(part2, ma_period, 1)) * 100
    return alpha


@factor_attr(max_depend=10, return_type="float")
def alpha191_024(data, ma_period=5, use_vwap: bool = True):
    """
    SMA(CLOSE-DELAY(CLOSE,5),5,1)
    """
    _p = "vwap" if use_vwap else "close"
    # alpha = (data[_p] - data[_p].shift(ma_period)).ewm(alpha=1 / ma_period).mean()
    alpha = SMA(data[_p] - data[_p].shift(ma_period), ma_period, 1)
    return alpha


@factor_attr(max_depend=300, return_type="float")
def alpha191_025(data, ma_period=7, vol_period=20, ret_period=150, use_vwap: bool = True):
    """
    (
        -1 * RANK((DELTA(CLOSE, 7) * (1 - RANK(DECAYLINEAR((VOLUME / MEAN(VOLUME, 20)), 9)))))
    ) *
    (1 + RANK(SUM(RET, 250)))
    todo: decaylinear
    """
    _p = "vwap" if use_vwap else "close"
    ret_sum = (data[_p] / data[_p].shift() - 1).rolling(ret_period).sum()
    part1 = -1 * data[_p].diff(ma_period).rank(axis=1, pct=True)
    part2 = (data["volume"] / data["volume"].rolling(vol_period).mean()).ewm(alpha=1 / 9).mean().rank(axis=1, pct=True)
    alpha = part1 * part2 * (1 + ret_sum.rank(axis=1, pct=True))
    return alpha


@factor_attr(max_depend=300, return_type="float")
def alpha191_026(data, ma_period=12, corr_period=200):
    """
    (MEAN(CLOSE, 7) - CLOSE) + CORR(VWAP, DELAY(CLOSE, 5), 230)
    """
    alpha = data["close"].rolling(ma_period).mean() - data["close"] + \
            data["vwap"].rolling(corr_period).corr(data["close"].shift(5))
    return alpha


@factor_attr(max_depend=30, return_type="float")
def alpha191_027(data, short_period=3, long_period=6, ma_period=12, use_vwap: bool = True):
    """
    WMA(
	    (CLOSE - DELAY(CLOSE, 3)) / DELAY(CLOSE, 3) * 100 + (CLOSE - DELAY(CLOSE, 6)) / DELAY(CLOSE, 6) * 100
	, 12)
	@res: 0.001434 -2.034917e-02
    """
    _p = "vwap" if use_vwap else "close"
    short = (data[_p] / data[_p].shift(short_period) - 1) * 100
    long = (data[_p] / data[_p].shift(long_period) - 1) * 100
    # alpha = (short + long).ewm(alpha=1 / ma_period).mean()
    alpha = SMA(short + long, ma_period, 1)
    return alpha


@factor_attr(max_depend=30, return_type="float")
def alpha191_028(data, period=9, ma_period=3):
    """
    3 * SMA((CLOSE - TSMIN(LOW, 9)) / (TSMAX(HIGH, 9) - TSMIN(LOW, 9)) * 100, 3, 1)
    -
    2 * SMA(SMA((CLOSE - TSMIN(LOW, 9)) / (MAX(HIGH, 9) - TSMAX(LOW, 9)) * 100, 3, 1), 3, 1)
    @res:
        with bug: -0.008942   -0.008950
        fixed:    0.003445   -0.007778
    """
    low_min = data["low"].rolling(period).min()
    high_max = data["high"].rolling(period).max()
    # alpha = 3 * ((data["close"] - low_min) / (high_max - low_min) * 100).ewm(alpha=1 / ma_period).mean()
    # alpha -= 2 * alpha.ewm(alpha=1 / ma_period).mean().ewm(alpha=1 / ma_period).mean()
    alpha = 3 * SMA((data["close"] - low_min) / (high_max - low_min) * 100, ma_period, 1) - \
            2 * SMA(SMA((data["close"] - low_min) / (high_max - low_min) * 100, ma_period, 1), ma_period, 1)
    return alpha


@factor_attr(max_depend=30, return_type="float")
def alpha191_029(data, use_vwap: bool = True):
    """
    (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*VOLUME
    """
    _p = "vwap" if use_vwap else "close"
    alpha = (data[_p] - data[_p].shift(6)) / data[_p].shift(6) * np.log(data["volume"])
    return alpha


@factor_attr(unfinished=True)
def alpha191_030(data):
    """
    WMA((REGRESI(CLOSE/DELAY(CLOSE)-1,MKT,SMB,HML， 60))^2,20)
    TODO
    """
    pass


@factor_attr(max_depend=30, return_type="float")
def alpha191_031(data, period=12, use_vwap: bool = True):
    """
    (CLOSE-MEAN(CLOSE,12))/MEAN(CLOSE,12)*100
    """
    _p = "vwap" if use_vwap else "close"
    alpha = (data[_p] - data[_p].rolling(period).mean()) / data[_p].rolling(period).mean() * 100
    return alpha


@factor_attr(max_depend=30, return_type="float")
def alpha191_032(data, period=3):
    """
    (-1 * SUM(RANK(CORR(RANK(HIGH), RANK(VOLUME), 3)), 3))
    @res: 0.003543  1.773410e-02
    """
    corr = data["high"].rank(axis=1, pct=True).rolling(period).corr(data["volume"].rank(axis=1, pct=True))
    alpha = -1 * corr.rank(axis=1, pct=True).rolling(period).sum()
    return alpha


@factor_attr(max_depend=300, return_type="float")
def alpha191_033(data, min_period=5):
    """
    (
        ((-1 * TSMIN(LOW, 5)) + DELAY(TSMIN(LOW, 5), 5)) * RANK(((SUM(RET, 240) - SUM(RET, 20)) / 220))
    ) * TSRANK(VOLUME, 5)
    @res: 0.003731  1.327557e-02
    """
    low_min = data["low"].rolling(min_period).min()
    ret = data["close"] / data["close"].shift() - 1
    ret_sum = ret.rolling(240).sum() - ret.rolling(20).sum()
    alpha = ((-1 * low_min) + low_min.shift(5)) * \
            ret_sum.rank(axis=1, pct=True) * \
            data["turn"].rank(axis=1, pct=True).shift(min_period)
    return alpha


@factor_attr(max_depend=30, return_type="float")
def alpha191_034(data, period=12, use_vwap: bool = True):
    """
    MEAN(CLOSE,12)/CLOSE
    """
    _p = "vwap" if use_vwap else "close"
    return data[_p].rolling(period).mean() / data[_p]


@factor_attr(max_depend=40, return_type="float")
def alpha191_035(data, ma_period1=15, ma_period2=7, corr_period=17):
    """
    MIN(
        RANK(DECAYLINEAR(DELTA(OPEN, 1), 15)),
        RANK(DECAYLINEAR(CORR((VOLUME), ((OPEN * 0.65) + (OPEN * 0.35)), 17), 7))
    ) * -1
    @res: 0.003040  1.670223e-02
    """
    part1 = (data["open"] - data["open"].shift(1)).ewm(alpha=1 / ma_period1).mean().rank(axis=1, pct=True)
    part2 = data["volume"].rolling(corr_period).corr((data["open"] * 0.65 + data["close"] * 0.35)).ewm(
        alpha=1 / ma_period2).mean().rank(axis=1, pct=True)
    alpha = np.minimum(part1, part2) * -1
    return alpha


@factor_attr(max_depend=30, return_type="float")
def alpha191_036(data, corr_period=6, sum_period=2):
    """
    RANK(SUM(CORR(RANK(VOLUME), RANK(VWAP)), 6), 2)
    """
    corr = data["volume"].rank(axis=1, pct=True).rolling(corr_period).corr(data["vwap"].rank(axis=1, pct=True))
    alpha = corr.rolling(sum_period).sum().rank(axis=1, pct=True)
    return alpha


@factor_attr(max_depend=30, return_type="float")
def alpha191_037(data, sum_period=5, delay_period=10, use_vwap: bool = True):
    """
    -1 * RANK(
        (SUM(OPEN, 5) * SUM(RET, 5)) - DELAY((SUM(OPEN, 5) * SUM(RET, 5)), 10)
    )
    @res: 0.002643    0.010871
    """
    _p = "vwap" if use_vwap else "close"
    ret = data[_p] / data[_p].shift() - 1
    alpha = data["open"].rolling(sum_period).sum() * ret.rolling(sum_period).sum()
    alpha = (alpha - alpha.shift(delay_period)).rank(axis=1, pct=True) * -1
    return alpha


@factor_attr(max_depend=30, return_type="float")
def alpha191_038(data, period=20):
    """
    (MEAN(HIGH, 20) < HIGH) ? (-1 * DELTA(HIGH, 2)) : 0
    @res:
    """
    condition = data["high"].rolling(period).mean() < data["high"]
    alpha = -1 * data["high"].diff(2)
    alpha[~condition] = 0
    return alpha


@factor_attr(max_depend=300, return_type="float")
def alpha191_039(data):
    """
    (
        RANK(DECAYLINEAR(DELTA((CLOSE), 2), 8)) - RANK(DECAYLINEAR(CORR(((VWAP * 0.3) + (OPEN * 0.7)), SUM(MEAN(VOLUME, 180), 37), 14), 12))
    ) * -1
    @res:
    """
    alpha = RANK(DECAYLINEAR(DELTA(data["close"], 2), 8))
    alpha -= RANK(DECAYLINEAR(CORR((data["vwap"] * 0.3 + data["open"] * 0.7),
                                   SUM(MEAN(data["volume"], 180), 37), 14), 12))
    return -1 * alpha


@factor_attr(max_depend=40, return_type="float")
def alpha191_040(data, period=26, use_vwap: bool = True):
    """
    SUM(
        (CLOSE > DELAY(CLOSE, 1) ? VOLUME : 0), 26
    ) /
    SUM(
        (CLOSE <= DELAY(CLOSE, 1) ? VOLUME : 0), 26
    ) * 100
    @res:
    """
    _p = "vwap" if use_vwap else "close"
    condition = data[_p] > data[_p].shift(1)
    part1 = data["volume"].where(condition, 0).rolling(period).sum()
    part2 = data["volume"].where(~condition, 0).rolling(period).sum()
    alpha = part1 / part2 * 100
    return alpha


@factor_attr(max_depend=30, return_type="float")
def alpha191_041(data, delta_period=3, max_period=5):
    """
    RANK(
        MAX(DELTA(VWAP, 3), 5)
    ) * -1
    @res:  0.003000  2.389447e-02
    """
    alpha = data["vwap"].diff(delta_period).rolling(max_period).max().rank(axis=1, pct=True) * -1
    return alpha


@factor_attr(max_depend=30, return_type="float")
def alpha191_042(data, period=10):
    """
    -1 * RANK(STD(HIGH, 10)) * CORR(HIGH, VOLUME, 10)
    @res: 1.015345e-02  3.029626e-02
    """
    alpha = -1 * data["high"].rolling(period).std().rank(axis=1, pct=True)
    alpha *= data["high"].rolling(period).corr(data["volume"])
    return alpha


@factor_attr(max_depend=30, return_type="float")
def alpha191_043(data, sum_period=6, use_vwap=True):
    """
    SUM(
	    (CLOSE > DELAY(CLOSE, 1) ? VOLUME : (CLOSE < DELAY(CLOSE, 1) ? -VOLUME : 0))
	, 6)
    @res: -0.001190 -1.507139e-02
    """
    _p = "vwap" if use_vwap else "close"
    condition = data[_p] > data[_p].shift(1)
    alpha = data["volume"].where(condition, -data["volume"])
    alpha = alpha.rolling(sum_period).sum()
    return alpha


@factor_attr(max_depend=40, return_type="float")
def alpha191_044(data):
    """
    (TSRANK(DECAYLINEAR(CORR(((LOW )), MEAN(VOLUME,10), 7), 6),4) + TSRANK(DECAYLINEAR(DELTA((VWAP),3), 10), 15))
    @res:
    """
    alpha = TS_RANK(DECAYLINEAR(CORR(data["low"], MEAN(data["volume"], 10), 7), 6), 4)
    alpha += TS_RANK(DECAYLINEAR(DELTA(data["vwap"], 3), 10), 15)
    return alpha


@factor_attr(max_depend=300, return_type="float")
def alpha191_045(data, corr_period=15, mean_period=150):
    """
    RANK(
        DELTA((CLOSE * 0.6 + OPEN * 0.4), 1)
    ) *
    RANK(
        CORR(VWAP, MEAN(VOLUME, 150), 15)
    )
    @res: -7.506140e-03 -1.240862e-02
    """
    alpha = (data["close"] * 0.6 + data["open"] * 0.4).diff(1).rank(axis=1, pct=True)
    alpha *= data["vwap"].rolling(corr_period).corr(data["volume"].rolling(mean_period).mean())
    return alpha


@factor_attr(max_depend=30, return_type="float")
def alpha191_046(data, min_window=3, use_vwap=False):
    """
    (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/(4*CLOSE)
    @res: 0.006388  2.398373e-02
    """
    _p = "vwap" if use_vwap else "close"
    alpha = data[_p].rolling(min_window).mean() + \
            data[_p].rolling(min_window * 2 ** 1).mean() + \
            data[_p].rolling(min_window * 2 ** 2).mean() + \
            data[_p].rolling(min_window * 2 ** 3).mean()
    alpha = alpha / (4 * data[_p])
    return alpha


@factor_attr(max_depend=30, return_type="float")
def alpha191_047(data, period=6, ma_period=9, use_vwap=False):
    """
    SMA(
	    (TSMAX(HIGH, 6) - CLOSE) / (TSMAX(HIGH, 6) - TSMIN(LOW, 6)) * 100
	, 9, 1)
    @res:   -0.010903    0.001250
    """
    _p = "vwap" if use_vwap else "close"
    alpha = (data["high"].rolling(period).max() - data[_p]) / \
            (data["high"].rolling(period).max() - data["low"].rolling(period).min()) * 100
    alpha = SMA(alpha, ma_period, 1)  # alpha.ewm(alpha=1 / ma_period).mean()
    return alpha


@factor_attr(max_depend=30, return_type="float")
def alpha191_048(data, use_vwap=False):
    """
    -1 * (
        (
            RANK(
                (
                    SIGN((CLOSE - DELAY(CLOSE, 1))) + SIGN((DELAY(CLOSE, 1) - DELAY(CLOSE, 2))) + SIGN((DELAY(CLOSE, 2) - DELAY(CLOSE, 3)))
                )
            )
        ) * SUM(VOLUME, 5)
    ) / SUM(VOLUME, 20)
    @res:
    """
    _p = "vwap" if use_vwap else "close"
    alpha = np.sign(data[_p].diff(1)) + \
            np.sign(data[_p].shift(1).diff(1)) + \
            np.sign(data[_p].shift(2).diff(1))
    alpha = alpha.rank(axis=1, pct=True) * data["volume"].rolling(5).sum() / data["volume"].rolling(20).sum()
    return alpha


@factor_attr(max_depend=30, return_type="float")
def alpha191_049(data, period=12):
    """
    condition = (HIGH + LOW) >= (DELAY(HIGH, 1) + DELAY(LOW, 1))
    low_diff = LOW - DELAY(LOW, 1)
    high_diff = HIGH - DELAY(HIGH, 1)
    SUM((condition ? 0 : MAX(ABS(high_diff), ABS(low_diff))), 12) /
    (
        SUM((condition ? 0 : MAX(ABS(high_diff), ABS(low_diff))), 12) +
        SUM((~condition ? 0 : MAX(ABS(high_diff), ABS(low_diff))), 12)
    )
    @res: -0.001646    0.014172
    """
    condition = (data["high"] + data["low"]) >= (data["high"].shift(1) + data["low"].shift(1))
    low_diff = np.abs(data["low"] - data["low"].shift(1))
    high_diff = np.abs(data["high"] - data["high"].shift(1))
    part = np.maximum(low_diff, high_diff)
    alpha = part.where(~condition, 0).rolling(period).sum() / \
            (part.where(~condition, 0).rolling(period).sum() + part.where(condition, 0).rolling(period).sum())
    return alpha


@factor_attr(max_depend=30, return_type="float")
def alpha191_050(data, period=12):
    """
    condition1 = (HIGH + LOW) <= (DELAY(HIGH, 1) + DELAY(LOW, 1))
    condition2 = (HIGH + LOW) >= (DELAY(HIGH, 1) + DELAY(LOW, 1))
    part = MAX(ABS(HIGH - DELAY(HIGH, 1)), ABS(LOW - DELAY(LOW, 1)))
    SUM((condition1 ? 0 : part), 12) / (SUM((condition1 ? 0 : part), 12) + SUM((condition2 ? 0 : part), 12)) -
    SUM((condition2 ? 0 : part), 12) / (SUM((condition2 ? 0 : part), 12) + SUM((condition1 ? 0 : part), 12))
    @res: 0.001650   -0.014326
    """
    condition1 = (data["high"] + data["low"]) <= (data["high"].shift(1) + data["low"].shift(1))
    condition2 = (data["high"] + data["low"]) >= (data["high"].shift(1) + data["low"].shift(1))
    part = np.maximum(np.abs(data["high"] - data["high"].shift(1)), np.abs(data["low"] - data["low"].shift(1)))
    part1 = part.where(~condition1, 0).rolling(period).sum()
    part2 = part.where(~condition2, 0).rolling(period).sum()
    alpha = (part1 - part2) / (part1 + part2 + 1e-7)
    return alpha


@factor_attr(max_depend=30, return_type="float")
def alpha191_051(data, period=12):
    """
    condition1 = (HIGH + LOW) <= (DELAY(HIGH, 1) + DELAY(LOW, 1))
    condition2 = (HIGH + LOW) >= (DELAY(HIGH, 1) + DELAY(LOW, 1))
    part = MAX(ABS(HIGH - DELAY(HIGH, 1)), ABS(LOW - DELAY(LOW, 1)))
    SUM((condition1 ? 0 : part), 12) / (SUM((condition1 ? 0 : part), 12) + SUM((condition2 ? 0 : part), 12))
    @res: 0.001650   -0.014330
    """
    condition1 = (data["high"] + data["low"]) <= (data["high"].shift(1) + data["low"].shift(1))
    condition2 = (data["high"] + data["low"]) >= (data["high"].shift(1) + data["low"].shift(1))
    part = np.maximum(np.abs(data["high"] - data["high"].shift(1)), np.abs(data["low"] - data["low"].shift(1)))
    part1 = part.where(~condition1, 0).rolling(period).sum()
    part2 = part.where(~condition2, 0).rolling(period).sum()
    alpha = part1 / (part1 + part2 + 1e-7)
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_052(data, period=26):
    """
    SUM(
        MAX(0, HIGH - DELAY((HIGH + LOW + CLOSE) / 3, 1)), 26
    ) /
    SUM(
        MAX(0, DELAY((HIGH + LOW + CLOSE) / 3, 1) - LOW), 26
    ) * 100
    @res: -0.003297 -2.125803e-02
    """
    delay_avg = ((data["high"] + data["low"] + data["close"]) / 3.).shift()
    alpha = (np.maximum(0, data["high"] - delay_avg)).rolling(period).sum() / \
            (np.maximum(0, delay_avg - data["low"])).rolling(period).sum() * 100
    return alpha


@factor_attr(max_depend=30, return_type="float")
def alpha191_053(data, period=12, use_vwap=False):
    """
    COUNT(CLOSE > DELAY(CLOSE, 1), 12) / 12 * 100
    @res: 0.003913   -0.003977
    """
    _p = "vwap" if use_vwap else "close"
    na_map = data[_p].where(data[_p].isna(), 0)
    alpha = (data[_p] > data[_p].shift(1)).astype(float).rolling(period).sum() / period * 100
    return alpha + na_map


@factor_attr(max_depend=30, return_type="float")
def alpha191_054(data, corr_period=10):
    """
    -1 * RANK(
        STD(ABS(CLOSE - OPEN) + (CLOSE - OPEN)) + CORR(CLOSE, OPEN,10)
    )
    @res:
    """
    alpha = (np.abs(data["close"] - data["open"]) + (data["close"] - data["open"])).rolling(corr_period).std() + \
            data["close"].rolling(corr_period).corr(data["open"])
    alpha = -alpha.rank(axis=1, pct=True)
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_055(data, period=20, use_vwap=False):
    """
    SUM(
        16 * (
            CLOSE - DELAY(CLOSE, 1) + (CLOSE - OPEN) / 2 + DELAY(CLOSE, 1) - DELAY(OPEN, 1)
        ) / (
            (ABS(HIGH - DELAY(CLOSE, 1)) > ABS(LOW - DELAY(CLOSE, 1)) & ABS(HIGH - DELAY(CLOSE, 1)) > ABS(HIGH - DELAY(LOW, 1)) ? ABS(HIGH - DELAY(CLOSE, 1)) + ABS(LOW - DELAY(CLOSE, 1)) / 2 + ABS(DELAY(CLOSE, 1) - DELAY(OPEN, 1)) / 4 : (ABS(LOW - DELAY(CLOSE, 1)) > ABS(HIGH - DELAY(LOW, 1)) & ABS(LOW - DELAY(CLOSE, 1)) > ABS(HIGH - DELAY(CLOSE, 1)) ? ABS(LOW - DELAY(CLOSE, 1)) + ABS(HIGH - DELAY(CLOSE, 1)) / 2 + ABS(DELAY(CLOSE, 1) - DELAY(OPEN, 1)) / 4 : ABS(HIGH - DELAY(LOW, 1)) + ABS(DELAY(CLOSE, 1) - DELAY(OPEN, 1)) / 4))
        ) * MAX(ABS(HIGH - DELAY(CLOSE, 1)), ABS(LOW - DELAY(CLOSE, 1)))
        , 20)
    -->
    part1 = ABS(HIGH - DELAY(CLOSE, 1));
    part2 = ABS(LOW - DELAY(CLOSE, 1));
    part3 = ABS(HIGH - DELAY(LOW, 1));
    part4 = ABS(DELAY(CLOSE, 1) - DELAY(OPEN, 1));
    SUM(
        16 * (CLOSE - DELAY(CLOSE, 1) + (CLOSE - OPEN) / 2 + DELAY(CLOSE, 1) - DELAY(OPEN, 1)) / (
            (part1 > part2 & part1 > part3 ? part1 + part2 / 2 + part4 / 4 : (part2 > part3 & part2 > part1 ? part2 + part1 / 2 + part4 / 4 : part3 + part4 / 4))
        ) * MAX(part1, part2)
        , 20)
    @res:
    """
    _p = "vwap" if use_vwap else "close"
    part1 = np.abs(data["high"] - DELAY(data[_p], 1))
    part2 = np.abs(data["low"] - DELAY(data[_p], 1))
    part3 = np.abs(data["high"] - DELAY(data["low"], 1))
    part4 = np.abs(DELAY(data[_p], 1) - DELAY(data["open"], 1))
    var1 = part1 + part2 / 2 + part4 / 4
    var2 = part2 + part1 / 2 + part4 / 4
    var3 = part3 + part4 / 4
    alpha = 16 * (data[_p] - DELAY(data[_p], 1) +
                  (data[_p] - data["open"]) / 2 +
                  DELAY(data[_p], 1) - DELAY(data["open"], 1))
    alpha = alpha / IFELSE((part1 > part2) & (part1 > part3), var1,
                           IFELSE((part2 > part3) & (part2 > part1), var2, var3))
    alpha = alpha * np.maximum(part1, part2)
    alpha = SUM(alpha, period)
    return alpha


@factor_attr(max_depend=100, return_type="float")
def alpha191_056(data):
    """
    RANK(OPEN - TSMIN(OPEN, 12)) <
    RANK((RANK(CORR(SUM(((HIGH + LOW) / 2), 19), SUM(MEAN(VOLUME, 40), 19), 13)) ^ 5))
    @res:
    """
    na_map = data["open"].isna().astype(float).replace(1., np.nan)
    alpha = RANK(data["open"] - TS_MIN(data["open"], 12)) < \
            RANK((RANK(CORR(SUM(((data["high"] + data["low"]) / 2), 19), SUM(MEAN(data["volume"], 40), 19), 13)) ** 5))
    return alpha + na_map


@factor_attr(max_depend=100, return_type="float")
def alpha191_057(data):
    """
    SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)
    @res: 0.005868   -0.003811
    """
    alpha = SMA((data["close"] - TS_MIN(data["low"], 9)) / \
                (TS_MAX(data["high"], 9) - TS_MIN(data["low"], 9)) * 100, 3, 1)
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_058(data, period=20, use_vwap=False):
    """
    COUNT(CLOSE>DELAY(CLOSE,1),20)/20*100
    @res: 0.003334   -0.007780
    """
    _p = "vwap" if use_vwap else "close"
    na_map = data["open"].isna().astype(float).replace(1., np.nan)
    alpha = COUNT(data[_p] > data[_p].shift(1), period, na_map=na_map) / period * 100
    return alpha + na_map


@factor_attr(max_depend=50, return_type="float")
def alpha191_059(data, sum_period=20, use_vwap=False):
    """
    SUM(
	(CLOSE = DELAY(CLOSE, 1) ?
		0 :
		CLOSE - (CLOSE > DELAY(CLOSE, 1) ?
			MIN(LOW, DELAY(CLOSE, 1)) :
			MAX(HIGH, DELAY(CLOSE, 1))))
	, 20)
    @res:  -0.007605 -2.259568e-02
    """
    _p = "vwap" if use_vwap else "close"
    condition = data[_p] > data[_p].shift(1)
    alpha = np.minimum(data["low"], data[_p].shift(1))
    alpha = alpha.where(condition, np.maximum(data["high"], data[_p].shift(1)))
    alpha = SUM(data[_p] - alpha, sum_period)
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_060(data, period=20):
    """
    SUM(
        ((CLOSE - LOW) - (HIGH - CLOSE)) /
        (HIGH - LOW) * VOLUME
	, 20)
    @res:   -0.005103   -0.008526
            -0.001746   -0.000207   # log v
    """
    alpha = SUM(((data["close"] - data["low"]) - (data["high"] - data["close"])) / \
                (data["high"] - data["low"]) * data["volume"], period)
    return alpha


@factor_attr(max_depend=100, return_type="float")
def alpha191_061(data):
    """
    MAX(RANK(DECAYLINEAR(DELTA(VWAP, 1), 12)),
    RANK(DECAYLINEAR(RANK(CORR((LOW), MEAN(VOLUME, 80), 8)), 17))) * -1
    @res:
    """
    alpha = np.maximum(
        RANK(DECAYLINEAR(DELTA(data["vwap"], 1), 12)),
        RANK(DECAYLINEAR(RANK(CORR((data["low"]), MEAN(data["volume"], 80), 8)), 17))
    )
    return alpha


@factor_attr(max_depend=30, return_type="float")
def alpha191_062(data, period=5):
    """
     (-1 * CORR(HIGH, RANK(VOLUME), 5))
    @res:   9.687078e-03  2.375355e-02
            9.948207e-03  2.392549e-02  # turn
    """
    alpha = -CORR(data["high"], RANK(data["turn"]), period)
    return alpha


@factor_attr(max_depend=30, return_type="float")
def alpha191_063(data, ma_period=6, use_vwap=True):
    """
    SMA(
        MAX(CLOSE - DELAY(CLOSE, 1), 0), 6, 1
    ) /
    SMA(
        ABS(CLOSE - DELAY(CLOSE, 1)), 6, 1
    ) * 100
    @res:   0.013111   -0.009398
    """
    _p = "vwap" if use_vwap else "close"
    part = data[_p] - data[_p].shift(1)
    alpha = SMA(np.maximum(part, 0), ma_period, 1) / \
            SMA(np.abs(part), ma_period, 1) * 100
    return alpha


@factor_attr(max_depend=100, return_type="float")
def alpha191_064(data):
    """
    MAX(
    RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 4), 4)),
    RANK(DECAYLINEAR(MAX(CORR(RANK(CLOSE), RANK(MEAN(VOLUME, 60)), 4), 13), 14))) * -1
    @res:
    """
    alpha = np.maximum(
        RANK(DECAYLINEAR(CORR(RANK(data["vwap"]), RANK(data["volume"]), 4), 4)),
        RANK(DECAYLINEAR(MAX(CORR(RANK(data["close"]), RANK(MEAN(data["volume"], 60)), 4), 13), 14))
    )
    return alpha


@factor_attr(max_depend=30, return_type="float")
def alpha191_065(data, period=6, use_vwap=False):
    """
    MEAN(CLOSE,6)/CLOSE
    @res:
    """
    _p = "vwap" if use_vwap else "close"
    alpha = MEAN(data[_p], period) / data[_p]
    return alpha


@factor_attr(max_depend=30, return_type="float")
def alpha191_066(data, period=6, use_vwap=False):
    """
    (CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)*100
    @res:
    """
    _p = "vwap" if use_vwap else "close"
    alpha = (data[_p] - MEAN(data[_p], period)) / MEAN(data[_p], period) * 100
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_067(data, period=24, use_vwap=False):
    """
    SMA(MAX(CLOSE-DELAY(CLOSE,1),0),24,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),24,1)*100
    @res:  0.010464   -0.015751
           0.012069   -0.013868 vwap
    """
    _p = "vwap" if use_vwap else "close"
    part = data[_p] - data[_p].shift(1)
    alpha = SMA(np.maximum(part, 0), period, 1) / \
            SMA(np.abs(part), period, 1) * 100
    return alpha


@factor_attr(max_depend=30, return_type="float")
def alpha191_068(data, period=15):
    """
    SMA(
        ((HIGH + LOW) / 2 - (DELAY(HIGH, 1) + DELAY(LOW, 1)) / 2) *
        (HIGH - LOW) / VOLUME
	, 15, 2)
    @res: -0.000878 -1.465395e-02
    """
    alpha = (data["high"] + data["low"]) / 2 - (data["high"].shift(1) + data["low"].shift(1)) / 2
    alpha = alpha * (data["high"] - data["low"]) / data["volume"]
    alpha = SMA(alpha, period, 2)
    return alpha


@factor_attr(max_depend=30, return_type="float")
def alpha191_069(data, period=20):
    """
    DTM = (OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1)))
    DBM = (OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1))))
    (SUM(DTM, 20) > SUM(DBM, 20) ? (SUM(DTM, 20) - SUM(DBM, 20)) / SUM(DTM, 20) : (SUM(DTM, 20) = SUM(DBM, 20) ? 0 : (SUM(DTM, 20) - SUM(DBM, 20)) / SUM(DBM, 20)))
    @res:
    """
    dtm = IFELSE(data["open"] <= DELAY(data["open"], 1), 0,
                 np.maximum(data["high"] - data["open"], data["open"] - DELAY(data["open"], 1)))
    dbm = IFELSE(data["open"] >= DELAY(data["open"], 1), 0,
                 np.maximum(data["open"] - data["low"], data["open"] - DELAY(data["open"], 1)))
    alpha = IFELSE(SUM(dtm, period) > SUM(dbm, period),
                   (SUM(dtm, period) - SUM(dbm, period)) / SUM(dtm, period),
                   IFELSE(SUM(dtm, period) == SUM(dbm, period), 0,
                          (SUM(dtm, period) - SUM(dbm, period)) / SUM(dbm, period)))
    return alpha


@factor_attr(max_depend=30, return_type="float")
def alpha191_070(data, period=6):
    """
    STD(AMOUNT,6)
    @res:
    """
    alpha = STD(data["amount"], period)
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_071(data, period=24, use_vwap=False):
    """
    (CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100
    @res:
    """
    _p = "vwap" if use_vwap else "close"
    alpha = (data[_p] - MEAN(data[_p], period)) / MEAN(data[_p], period) * 100
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_072(data, minmax_period=6, ma_period=15):
    """
    SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,15,1)
    @res:
    """
    alpha = (TS_MAX(data["high"], minmax_period) - data["close"]) / \
            (TS_MAX(data["high"], minmax_period) - TS_MIN(data["low"], minmax_period)) * 100
    alpha = SMA(alpha, ma_period, 1)
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_073(data):
    """
    -1 * TSRANK(DECAYLINEAR(DECAYLINEAR(CORR((CLOSE), VOLUME, 10), 16), 4), 5) -
    RANK(DECAYLINEAR(CORR(VWAP, MEAN(VOLUME, 30), 4), 3))
    @res:
    """
    alpha = -1 * TS_RANK(DECAYLINEAR(DECAYLINEAR(CORR(data["close"], data["volume"], 10), 16), 4), 5) - \
            RANK(DECAYLINEAR(CORR(data["vwap"], MEAN(data["volume"], 30), 4), 3))
    return alpha


@factor_attr(max_depend=100, return_type="float")
def alpha191_074(data, sum_period=20, corr_period1=7, corr_period2=6):
    """
    RANK(
        CORR(SUM(((LOW * 0.35) + (VWAP * 0.65)), 20), SUM(MEAN(VOLUME, 40), 20), 7)
    )  + RANK(CORR(RANK(VWAP), RANK(VOLUME), 6))
    @res:
    """
    alpha = RANK(CORR(SUM(data["low"] * 0.35 + data["vwap"] * 0.65, sum_period),
                      SUM(MEAN(data["volume"], 40), sum_period), corr_period1)
                 ) + RANK(CORR(RANK(data["vwap"]), RANK(data["volume"]), corr_period2))
    return alpha


@factor_attr(max_depend=100, return_type="float")
def alpha191_075(data, period=50):
    """
    COUNT(CLOSE > OPEN & BANCHMARKINDEXCLOSE < BANCHMARKINDEXOPEN, 50) / COUNT(BANCHMARKINDEXCLOSE < BANCHMARKINDEXOPEN, 50)
    @res:
    """
    bench = (data["close"] / DELAY(data["close"], 1) - 1).mean(axis=1)
    na_map = data["close"].isna().astype(float).replace(1., np.nan)
    alpha = COUNT((data["close"] > data["open"]).mul(bench < 0, axis="index"), period, na_map=na_map) / \
            COUNT((data["close"] != data["open"]).mul(bench < 0, axis="index"), period, na_map=na_map)
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_076(data, period=20, use_vwap=False):
    """
    STD(ABS((CLOSE / DELAY(CLOSE, 1) - 1)) / VOLUME, 20) /
    MEAN(ABS((CLOSE / DELAY(CLOSE, 1) - 1)) / VOLUME, 20)
    @res:
    """
    _p = "vwap" if use_vwap else "close"
    alpha = STD(np.abs((data[_p] / data[_p].shift(1) - 1) / data["volume"]), period) / \
            MEAN(np.abs((data[_p] / data[_p].shift(1) - 1) / data["volume"]), period)
    return alpha


@factor_attr(max_depend=60, return_type="float")
def alpha191_077(data):
    """
    MIN(
    RANK(DECAYLINEAR(((((HIGH + LOW) / 2) + HIGH) - (VWAP + HIGH)), 20)),
    RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME, 40), 3), 6)))
    @res:
    """
    alpha = RANK(DECAYLINEAR((((data["high"] + data["low"]) / 2 + data["high"]) - (data["vwap"] + data["high"])), 20))
    alpha = np.minimum(alpha, RANK(DECAYLINEAR(CORR(((data["high"] + data["low"]) / 2),
                                                    MEAN(data["volume"], 40), 3), 6)))
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_078(data, period=12, use_vwap=False):
    """
    ((HIGH + LOW + CLOSE) / 3 - MA((HIGH + LOW + CLOSE) / 3, 12)) /
    (0.015 * MEAN(ABS(CLOSE - MEAN((HIGH + LOW + CLOSE) / 3, 12)), 12))
    @res:
    """
    _p = "vwap" if use_vwap else "close"
    alpha = ((data["high"] + data["low"] + data[_p]) / 3 - MEAN((data["high"] + data["low"] + data[_p]) / 3, period)) / \
            (0.015 * MEAN(np.abs(data[_p] - MEAN((data["high"] + data["low"] + data[_p]) / 3, period)), period))
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_079(data, period=12, use_vwap=False):
    """
    SMA(MAX(CLOSE - DELAY(CLOSE, 1), 0), 12, 1) /
    SMA(ABS(CLOSE - DELAY(CLOSE, 1)), 12, 1) * 100
    @res:
    """
    _p = "vwap" if use_vwap else "close"
    alpha = SMA(np.maximum(data[_p] - data[_p].shift(1), 0), period, 1) / \
            SMA(np.abs(data[_p] - data[_p].shift(1)), period, 1) * 100
    return alpha


@factor_attr(max_depend=30, return_type="float")
def alpha191_080(data, period=5):
    """
    (VOLUME-DELAY(VOLUME,5))/DELAY(VOLUME,5)*100
    @res:
    """
    alpha = (data["volume"] - data["volume"].shift(period)) / data["volume"].shift(period) * 100
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_081(data, long_period=21, short_period=2):
    """
    SMA(VOLUME,21,2)
    @res:
    """
    alpha = SMA(data["volume"], long_period, short_period)
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_082(data, minmax_period=6, ma_period=20, use_vwap=False):
    """
    SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,20,1)
    @res:
    """
    _p = "vwap" if use_vwap else "close"
    alpha = (TS_MAX(data["high"], minmax_period) - data[_p]) / \
            (TS_MAX(data["high"], minmax_period) - TS_MIN(data["low"], minmax_period)) * 100
    alpha = SMA(alpha, ma_period, 1)
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_083(data, cov_period=5):
    """
    (-1 * RANK(COVIANCE(RANK(HIGH), RANK(VOLUME), 5)))
    @res:   0.003973  2.304484e-02
    """
    alpha = -1 * RANK(COVARIANCE(RANK(data["high"]), RANK(data["volume"]), cov_period))
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_084(data, ma_period=20, use_vwap=False):
    """
    SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),20)
    @res:
    """
    _p = "vwap" if use_vwap else "close"
    alpha = data["volume"]
    alpha = alpha.where(data[_p] > data[_p].shift(1), -data["volume"])
    alpha = SUM(alpha, ma_period)
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_085(data, long_period=20, short_period=8, use_vwap=False):
    """
    TSRANK((VOLUME / MEAN(VOLUME, 20)), 20) *
    TSRANK((-1 * DELTA(CLOSE, 7)), 8)
    @res: 0.005722    0.006362
    """
    _p = "vwap" if use_vwap else "close"
    alpha = TS_RANK(data["volume"] / MEAN(data["volume"], long_period), long_period) * \
            TS_RANK(-1 * DELTA(data[_p], short_period), short_period)
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_086(data, long_period=20, short_period=8, use_vwap=False):
    """
    part1 = ((DELAY(CLOSE, 20) - DELAY(CLOSE, 10)) / 10)
    part2 = ((DELAY(CLOSE, 10) - CLOSE) / 10)
    (
        (0.25 < (part1 - part2))
            ? -1 : (((part1 - part2) < 0)
                ? 1 : (-1 * (CLOSE - DELAY(CLOSE, 1))))
    )
    todo check
    @res: 0.002384    -0.000863
    """
    _p = "vwap" if use_vwap else "close"
    na_map = data[_p].isna().astype(float).replace(1., np.nan)
    part1 = (data[_p].shift(long_period) - data[_p].shift(short_period)) / short_period
    part2 = (data[_p].shift(short_period) - data[_p]) / short_period
    alpha = -1 * (data[_p] - data[_p].shift(1))
    alpha = alpha.where(part1 - part2 >= 0.25, -1)
    alpha = alpha.where(part1 - part2 > 0, 1)
    return alpha + na_map


@factor_attr(max_depend=30, return_type="float")
def alpha191_087(data, period=7):
    """
    (RANK(DECAYLINEAR(DELTA(VWAP, 4), 7)) +
        TSRANK(DECAYLINEAR(((((LOW * 0.9) + (LOW * 0.1)) - VWAP) / (OPEN - ((HIGH + LOW) / 2))), 11), 7))
    * -1
    @res:
    """
    alpha = (RANK(DECAYLINEAR(DELTA(data["vwap"], 4), period)) +
             TS_RANK(DECAYLINEAR(((((data["low"] * 0.9) + (data["low"] * 0.1)) - data["vwap"]) /
                                  (data["open"] - ((data["high"] + data["low"]) / 2) + 1e-7)), 11), period)) * -1
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_088(data, period=20, use_vwap=False):
    """
    (CLOSE-DELAY(CLOSE,20))/DELAY(CLOSE,20)*100
    @res:
    """
    _p = "vwap" if use_vwap else "close"
    alpha = (data[_p] - data[_p].shift(period)) / data[_p].shift(period) * 100
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_089(data, long_period=27, short_period=13, final_period=10, use_vwap=False):
    """
    2 * (
        SMA(CLOSE, 13, 2) - SMA(CLOSE, 27, 2) -
        SMA(
            SMA(CLOSE, 13, 2) - SMA(CLOSE, 27, 2)
            , 10, 2)
    )
    @res:
    """
    _p = "vwap" if use_vwap else "close"
    ma_short = SMA(data[_p], short_period, 2)
    ma_long = SMA(data[_p], long_period, 2)
    alpha = 2 * (ma_short - ma_long - SMA(ma_short - ma_long, final_period, 2))
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_090(data, corr_period=5):
    """
    RANK(CORR(RANK(VWAP), RANK(VOLUME), 5)) * -1
    @res:
    """
    alpha = RANK(CORR(RANK(data["vwap"]), RANK(data["volume"]), corr_period)) * -1
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_091(data, vol_period=40, corr_period=5, use_vwap=False):
    """
    (
        RANK((CLOSE - MAX(CLOSE, 5))) *
        RANK(CORR(MEAN(VOLUME, 40), LOW, 5))
    ) * -1
    @res:
    @NOTE:
        MAX(CLOSE, 5) ???
    """
    _p = "vwap" if use_vwap else "close"
    alpha = RANK(data[_p] - TS_MAX(data[_p], 5)) * \
            RANK(CORR(MEAN(data["volume"], vol_period), data["low"], corr_period)) * -1
    return alpha


@factor_attr(max_depend=300, return_type="float")
def alpha191_092(data, decay_period1=3, decay_period2=5, rank_period=15):
    """
    MAX(
        RANK(DECAYLINEAR(DELTA(((CLOSE * 0.35) + (VWAP * 0.65)), 2), 3)),
        TSRANK(DECAYLINEAR(ABS(CORR((MEAN(VOLUME, 180)), CLOSE, 13)), 5), 15)
    ) * -1
    @res:
    """
    alpha = MAX(
        RANK(DECAYLINEAR(DELTA((data["close"] * 0.35 + data["vwap"] * 0.65), 2), decay_period1)),
        TS_RANK(DECAYLINEAR(np.abs(CORR(MEAN(data["volume"], 180), data["close"], 13)), decay_period2), rank_period)
    ) * -1
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_093(data, sum_period=20):
    """
    SUM((OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1)))),20)
    @res:
    """
    alpha = MAX(data["open"] - data["low"], data["open"] - data["open"].shift(1))
    alpha = SUM(IFELSE(data["open"] >= data["open"].shift(1), 0, alpha), sum_period)
    # SUM(alpha.where(data["open"] < data["open"].shift(1), 0), sum_period)
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_094(data, period=30, use_vwap=False):
    """
    SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),30)
    @res:
    """
    _p = "vwap" if use_vwap else "close"
    alpha = IFELSE(data[_p] < data[_p].shift(1), -data["volume"], 0)
    alpha = IFELSE(data[_p] > data[_p].shift(1), data["volume"], alpha)
    alpha = SUM(alpha, period)
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_095(data, period=20):
    """
    STD(AMOUNT,20)
    @res:
    """
    alpha = STD(data["amount"], period)
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_096(data, minmax_period=9, ma_period=3):
    """
    SMA(
        SMA(
            (CLOSE - TSMIN(LOW, 9)) /
            (TSMAX(HIGH, 9) - TSMIN(LOW, 9)) * 100, 3, 1), 3, 1)
    @res:
    """
    alpha = SMA(SMA(
        (data["close"] - TS_MIN(data["low"], minmax_period)) /
        (TS_MAX(data["high"], minmax_period) - TS_MIN(data["low"], minmax_period)) * 100,
        ma_period, 1), ma_period, 1)
    return alpha


@factor_attr(max_depend=30, return_type="float")
def alpha191_097(data, period=10):
    """
    STD(VOLUME,10)
    @res:
    """
    alpha = STD(data["volume"], period)
    return alpha


@factor_attr(max_depend=200, return_type="float")
def alpha191_098(data, period=100, use_vwap=False):
    """
    ((((DELTA((SUM(CLOSE, 100) / 100), 100) / DELAY(CLOSE, 100)) < 0.05) ||
	((DELTA((SUM(CLOSE, 100) / 100), 100) / DELAY(CLOSE, 100)) == 0.05)) ? (-1 * (CLOSE - TSMIN(CLOSE, 100))) : (-1 * DELTA(CLOSE, 3)))
	-->
	part1 = DELTA((SUM(CLOSE, 100) / 100), 100) / DELAY(CLOSE, 100)
    ((
        (part1 <= 0.05) ? (-1 * (CLOSE - TSMIN(CLOSE, 100))) : (-1 * DELTA(CLOSE, 3)))
    @res:
    """
    _p = "vwap" if use_vwap else "close"
    condition = DELTA(SUM(data[_p], period) / period, period) / data[_p].shift(period)
    alpha = IFELSE(condition <= 0.05, -1 * (data[_p] - TS_MIN(data[_p], period)), -1 * DELTA(data[_p], 3))
    return alpha


@factor_attr(max_depend=30, return_type="float")
def alpha191_099(data, cov_period=5, use_vwap=False):
    """
    (-1 * RANK(COVARIANCE(RANK(CLOSE), RANK(VOLUME), 5)))
    @res:
    """
    _p = "vwap" if use_vwap else "close"
    alpha = -1 * RANK(COVARIANCE(RANK(data[_p]), RANK(data["volume"]), cov_period))
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_100(data, period=20):
    """
    STD(VOLUME,20)
    @res:
    """
    alpha = STD(data["volume"], period)
    return alpha


@factor_attr(max_depend=100, return_type="binary")
def alpha191_101(data, corr_period1=15, corr_period2=11, ma_period1=30, ma_period2=37):
    """
    ((RANK(CORR(CLOSE, SUM(MEAN(VOLUME, 30), 37), 15)) < RANK(CORR(RANK(((HIGH * 0.1) + (VWAP * 0.9))), RANK(VOLUME), 11))) * -1)
    @res:
    """
    alpha = (RANK(CORR(data["close"], SUM(MEAN(data["volume"], ma_period1), ma_period2), corr_period1)) <
             RANK(CORR(RANK((data["high"] * 0.1 + data["vwap"] * 0.9)), RANK(data["volume"]), corr_period2))) * -1
    return alpha


@factor_attr(max_depend=30, return_type="float")
def alpha191_102(data, period=6):
    """
    SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)*100
    @res:
    """
    alpha = SMA(MAX(data["volume"] - DELAY(data["volume"], 1), 0), period, 1) / \
            SMA(np.abs(data["volume"] - DELAY(data["volume"], 1)), period, 1) * 100
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_103(data, period=20):
    """
    ((20-LOWDAY(LOW,20))/20)*100
    #TODO: LOWDAY CHECK
    @res:
    """
    alpha = (period - LOWDAY(data["low"], period)) / 20 * 100
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_104(data, corr_period=5, std_period=20, use_vwap=False):
    """
    (-1 * (DELTA(CORR(HIGH, VOLUME, 5), 5) * RANK(STD(CLOSE, 20))))
    @res:
    """
    _p = "vwap" if use_vwap else "close"
    alpha = -1 * (DELTA(CORR(data["high"], data["volume"], corr_period), corr_period) * RANK(STD(data[_p], std_period)))
    return alpha


@factor_attr(max_depend=30, return_type="float")
def alpha191_105(data, period=10):
    """
    (-1 * CORR(RANK(OPEN), RANK(VOLUME), 10))
    @res:
    """
    alpha = -1 * CORR(RANK(data["open"]), RANK(data["volume"]), period)
    return alpha


@factor_attr(max_depend=30, return_type="float")
def alpha191_106(data, period=20, use_vwap=False):
    """
    CLOSE-DELAY(CLOSE,20)
    @res:
    """
    _p = "vwap" if use_vwap else "close"
    alpha = data[_p] - DELAY(data[_p], period)
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_107(data):
    """
    (((-1 * RANK((OPEN - DELAY(HIGH, 1)))) * RANK((OPEN - DELAY(CLOSE, 1)))) * RANK((OPEN - DELAY(LOW, 1))))
    @res:
    """
    alpha = -1 * RANK((data["open"] - DELAY(data["high"], 1))) * \
            RANK((data["open"] - DELAY(data["close"], 1))) * \
            RANK((data["open"] - DELAY(data["low"], 1)))
    return alpha


@factor_attr(max_depend=200, return_type="float")
def alpha191_108(data):
    """
    ((RANK((HIGH - MIN(HIGH, 2)))^RANK(CORR((VWAP), (MEAN(VOLUME,120)), 6))) * -1)
    @res:
    """
    alpha = (RANK((data["high"] - TS_MIN(data["high"], 2))) ** RANK(
        CORR((data["vwap"]), (MEAN(data["volume"], 120)), 6)
    )) * -1
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_109(data, period=10, min_period=2):
    """
    SMA(HIGH-LOW,10,2)/SMA(SMA(HIGH-LOW,10,2),10,2)
    @res:
    """
    alpha = SMA(data["high"] - data["low"], period, min_period) / \
            SMA(SMA(data["high"] - data["low"], period, min_period), period, min_period)
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_110(data, period=20):
    """
    SUM(MAX(0,HIGH-DELAY(CLOSE,1)),20)/SUM(MAX(0,DELAY(CLOSE,1)-LOW),20)*100
    @res:
    """
    alpha = SUM(MAX(data["high"] - DELAY(data["close"], 1), 0), period) / \
            SUM(MAX(DELAY(data["close"], 1) - data["low"], 0), period) * 100
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_111(data, period1=11, period2=4):
    """
    SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),11,2)-SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),4,2)
    @res:
    """
    part = ((data["close"] - data["low"]) - (data["high"] - data["close"])) / (data["high"] - data["low"])
    alpha = SMA(data["volume"] * part, period1, 2) - \
            SMA(data["volume"] * part, period2, 2)
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_112(data, period=12, use_vwap=False):
    """
    (
        SUM((CLOSE - DELAY(CLOSE, 1) > 0 ? CLOSE - DELAY(CLOSE, 1) : 0), 12) -
        SUM((CLOSE - DELAY(CLOSE, 1) < 0 ? ABS(CLOSE - DELAY(CLOSE, 1)) : 0), 12)) /
    (
        SUM((CLOSE - DELAY(CLOSE, 1) > 0 ? CLOSE - DELAY(CLOSE, 1) : 0), 12) +
        SUM((CLOSE - DELAY(CLOSE, 1) < 0 ? ABS(CLOSE - DELAY(CLOSE, 1)) : 0), 12)) * 100
    @res:
    """
    _p = "vwap" if use_vwap else "close"
    alpha = (SUM(MAX(data[_p] - DELAY(data[_p], 1), 0), period) -
             SUM(MAX(DELAY(data[_p], 1) - data[_p], 0), period)) / \
            (SUM(MAX(data[_p] - DELAY(data[_p], 1), 0), period) +
             SUM(MAX(DELAY(data[_p], 1) - data[_p], 0), period)) * 100
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_113(data, period=20, corr_period=20, use_vwap=False):
    """
    -1 * (
        (RANK((SUM(DELAY(CLOSE, 5), 20) / 20)) * CORR(CLOSE, VOLUME, 2)) *
        RANK(CORR(SUM(CLOSE, 5), SUM(CLOSE, 20), 2))
    )
    @res:
    """
    _p = "vwap" if use_vwap else "close"
    alpha = -1 * (RANK(SUM(DELAY(data[_p], 5), period) / period) *
                  CORR(data[_p], data["volume"], corr_period) *
                  RANK(CORR(SUM(data[_p], 5), SUM(data[_p], period), corr_period)))
    return alpha


@factor_attr(max_depend=30, return_type="float")
def alpha191_114(data, period=5):
    """
    (RANK(DELAY(((HIGH - LOW) / (SUM(CLOSE, 5) / 5)), 2)) * RANK(RANK(VOLUME))) /
    (((HIGH - LOW) / (SUM(CLOSE, 5) / 5)) / (VWAP - CLOSE))
    @res:
    """
    part = (data["high"] - data["low"]) / MEAN(data["close"], period)
    alpha = (RANK(DELAY(part, 2)) * RANK(RANK(data["volume"]))) / \
            (part / (data["vwap"] - data["close"] + 1e-7))
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_115(data):
    """
    RANK(CORR(((HIGH * 0.9) + (CLOSE * 0.1)), MEAN(VOLUME, 30), 10)) ^
    RANK(CORR(TSRANK(((HIGH + LOW) / 2), 4), TSRANK(VOLUME, 10), 7))
    @res:
    """
    alpha = (RANK(CORR((data["high"] * 0.9 + data["close"] * 0.1), MEAN(data["volume"], 30), 10)) **
             RANK(CORR(TS_RANK((data["high"] + data["low"]) / 2, 4), TS_RANK(data["volume"], 10), 7)))
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_116(data, period=20, use_vwap=False):
    """
    REGBETA(CLOSE,SEQUENCE,20)
    @res:
    """
    _p = "vwap" if use_vwap else "close"
    idx = data[_p].index
    alpha = data[_p].apply(lambda x: pd.Series(rolling_slope(x.values, period)))
    alpha.index = idx
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_117(data, period1=16, period2=32):
    """
    (TSRANK(VOLUME, 32) * (1 - TSRANK(((CLOSE + HIGH) - LOW), 16))) * (1 - TSRANK(RET, 32))
    @res:
    """
    ret = data["close"] / DELAY(data["close"], 1) - 1
    alpha = (TS_RANK(data["volume"], period2) * (1 - TS_RANK((data["close"] + data["high"]) - data["low"], period1))) * \
            (1 - TS_RANK(ret, period2))
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_118(data, period=20):
    """
    SUM(HIGH-OPEN,20)/SUM(OPEN-LOW,20)*100
    @res:
    """
    alpha = SUM(data["high"] - data["open"], period) / SUM(data["open"] - data["low"], period) * 100
    return alpha


@factor_attr(max_depend=100, return_type="float")
def alpha191_119(data):
    """
    RANK(DECAYLINEAR(CORR(VWAP, SUM(MEAN(VOLUME, 5), 26), 5), 7)) -
    RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(OPEN), RANK(MEAN(VOLUME, 15)), 21), 9), 7), 8))
    @res:
    """
    alpha = RANK(DECAYLINEAR(CORR(data["vwap"], SUM(MEAN(data["volume"], 5), 26), 5), 7)) - \
            RANK(DECAYLINEAR(TS_RANK(TS_MIN(CORR(RANK(data["open"]), RANK(MEAN(data["volume"], 15)), 21), 9), 7), 8))
    return alpha


@factor_attr(max_depend=30, return_type="float")
def alpha191_120(data):
    """
    (RANK((VWAP - CLOSE)) / RANK((VWAP + CLOSE)))
    @res:
    """
    alpha = RANK(data["vwap"] - data["close"]) / RANK(data["vwap"] + data["close"])
    return alpha


@factor_attr(max_depend=100, return_type="float")
def alpha191_121(data):
    """
    (RANK((VWAP - MIN(VWAP, 12))) ^ TSRANK(CORR(TSRANK(VWAP, 20), TSRANK(MEAN(VOLUME, 60), 2), 18), 3)) * -1
    @res:
    """
    alpha = (RANK(data["vwap"] - TS_MIN(data["vwap"], 12)) ** TS_RANK(
        CORR(TS_RANK(data["vwap"], 20), TS_RANK(MEAN(data["volume"], 60), 2), 18), 3)) * -1
    return alpha


@factor_attr(max_depend=100, return_type="float")
def alpha191_122(data, period=13, use_vwap=False):
    """
    (SMA(SMA(SMA(LOG(CLOSE), 13, 2), 13, 2), 13, 2) - DELAY(SMA(SMA(SMA(LOG(CLOSE), 13, 2), 13, 2), 13, 2), 1)) /
    DELAY(SMA(SMA(SMA(LOG(CLOSE), 13, 2), 13, 2), 13, 2), 1)

    @res:
    """
    _p = "vwap" if use_vwap else "close"
    part = SMA(SMA(SMA(np.log(data[_p]), period, 2), period, 2), period, 2)
    alpha = (part - DELAY(part, 1)) / DELAY(part, 1)
    return alpha


@factor_attr(max_depend=100, return_type="binary")
def alpha191_123(data, period=20):
    """
    (RANK(CORR(SUM(((HIGH + LOW) / 2), 20), SUM(MEAN(VOLUME, 60), 20), 9)) < RANK(CORR(LOW, VOLUME, 6))) * -1
    @res:
    """
    alpha = RANK(CORR(SUM((data["high"] + data["low"]) / 2, period), SUM(MEAN(data["volume"], 60), period), 9))
    na_map = alpha.isna().astype(float).replace(1., np.nan)
    alpha = (alpha < RANK(CORR(data["low"], data["volume"], 6))).replace(False, -1) * -1
    return alpha + na_map


@factor_attr(max_depend=50, return_type="float")
def alpha191_124(data, period=30):
    """
    (CLOSE - VWAP) / DECAYLINEAR(RANK(TSMAX(CLOSE, 30)),2)
    @res:
    """
    alpha = (data["close"] - data["vwap"]) / DECAYLINEAR(RANK(TS_MAX(data["close"], period)), 2)
    return alpha


@factor_attr(max_depend=200, return_type="float")
def alpha191_125(data):
    """
    RANK(DECAYLINEAR(CORR((VWAP), MEAN(VOLUME, 80), 17), 20)) /
    RANK(DECAYLINEAR(DELTA(((CLOSE * 0.5) + (VWAP * 0.5)), 3), 16))
    @res:
    """
    alpha = RANK(DECAYLINEAR(CORR(data["vwap"], MEAN(data["volume"], 80), 17), 20)) / \
            RANK(DECAYLINEAR(DELTA((data["close"] * 0.5 + data["vwap"] * 0.5), 3), 16))
    return alpha


@factor_attr(max_depend=10, return_type="float")
def alpha191_126(data):
    """
     (CLOSE+HIGH+LOW)/3
    @res:
    """
    alpha = (data["close"] + data["high"] + data["low"]) / 3
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_127(data, period=12):
    """
    (MEAN((100*(CLOSE-MAX(CLOSE,12))/(MAX(CLOSE,12)))^2))^(1/2)
    @res:
    """
    alpha = (MEAN((100 * (data["close"] - TS_MAX(data["close"], period)) / (TS_MAX(data["close"], period))) ** 2,
                  period)) ** 0.5
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_128(data, period=14):
    """
    100 - (
    100 /
        (1 + SUM(((HIGH + LOW + CLOSE) / 3 > DELAY((HIGH + LOW + CLOSE) / 3, 1) ? (HIGH + LOW + CLOSE) / 3 * VOLUME : 0), 14) /
            SUM(((HIGH + LOW + CLOSE) / 3 < DELAY((HIGH + LOW + CLOSE) / 3, 1) ? (HIGH + LOW + CLOSE) / 3 * VOLUME : 0), 14))
    )
    @res:
    """
    part = ((data["high"] + data["low"] + data["close"]) / 3)
    condition = (part > DELAY(part, 1))
    alpha = SUM(IFELSE(condition, part * data["volume"], 0), period) / \
            (SUM(IFELSE(~condition, part * data["volume"], 0), period) + 1e-7)
    alpha = 100 - (100 / (1 + alpha))
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_129(data, period=12, use_vwap=False):
    """
    SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12)
    @res:
    """
    _p = "vwap" if use_vwap else "close"
    part = data[_p] - DELAY(data[_p], 1)
    alpha = SUM(IFELSE(part < 0, np.abs(part), 0), period)
    return alpha


@factor_attr(max_depend=100, return_type="float")
def alpha191_130(data):
    """
    RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME, 40), 9), 10)) /
    RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 7), 3))
    @res:
    """
    alpha = RANK(DECAYLINEAR(CORR((data["high"] + data["low"]) / 2, MEAN(data["volume"], 40), 9), 10)) / \
            RANK(DECAYLINEAR(CORR(RANK(data["vwap"]), RANK(data["volume"]), 7), 3))
    return alpha


@factor_attr(max_depend=100, return_type="float")
def alpha191_131(data, period=18, vol_period=50):
    """
     (RANK(DELAT(VWAP, 1))^TSRANK(CORR(CLOSE,MEAN(VOLUME,50), 18), 18))
    @res:
    """
    alpha = (RANK(DELTA(data["vwap"], 1)) **
             TS_RANK(CORR(data["close"], MEAN(data["volume"], vol_period), period), period))
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_132(data, period=20):
    """
    MEAN(AMOUNT,20)
    @res:
    """
    alpha = MEAN(data["amount"], period)
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_133(data, period=20):
    """
    ((20-HIGHDAY(HIGH,20))/20)*100-((20-LOWDAY(LOW,20))/20)*100
    @res:
    """
    alpha = ((20 - HIGHDAY(data["high"], period)) / 20) * 100 - \
            ((20 - LOWDAY(data["low"], period)) / 20) * 100
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_134(data, period=12, use_vwap=False):
    """
    (CLOSE-DELAY(CLOSE,12))/DELAY(CLOSE,12)*VOLUME
    @res:
    """
    _p = "vwap" if use_vwap else "close"
    alpha = (data[_p] - DELAY(data[_p], period)) / DELAY(data[_p], period) * data["volume"]
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_135(data, period=12, use_vwap=False):
    """
    SMA(DELAY(CLOSE/DELAY(CLOSE,20),1),20,1)
    @res:
    """
    _p = "vwap" if use_vwap else "close"
    alpha = SMA(DELAY(data[_p] / DELAY(data[_p], period), 1), period, 1)
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_136(data, corr_period=10):
    """
    ((-1 * RANK(DELTA(RET, 3))) * CORR(OPEN, VOLUME, 10))
    @res:
    """
    ret = data["close"] / DELAY(data["close"], 1) - 1
    alpha = (-1 * RANK(DELTA(ret, 3))) * CORR(data["open"], data["volume"], corr_period)
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_137(data, use_vwap=False):
    """
    condition1 = abshc > abslc;
    condition2 = abshc > abshl;
    condition3 = abslc > abshc;
    abshc = ABS(HIGH - DELAY(CLOSE, 1));
    abslc = ABS(LOW - DELAY(CLOSE, 1));
    absco = ABS(DELAY(CLOSE, 1) - DELAY(OPEN, 1));
    abshl = ABS(HIGH - DELAY(LOW, 1));
    16 * (CLOSE - DELAY(CLOSE, 1) + (CLOSE - OPEN) / 2 + DELAY(CLOSE, 1) - DELAY(OPEN, 1)) /
    ((condition1 & condition2 ? abshc + abslc / 2 + absco / 4 : (abslc > abshl & condition3 ? abslc + abshc / 2 + absco / 4 : abshl + absco / 4)))
    * MAX(abshc, abslc)
    @res:
    """
    _p = "vwap" if use_vwap else "close"
    abshc = np.abs(data["high"] - DELAY(data[_p], 1))
    abslc = np.abs(data["low"] - DELAY(data[_p], 1))
    absco = np.abs(DELAY(data[_p], 1) - DELAY(data["open"], 1))
    abshl = np.abs(data["high"] - DELAY(data["low"], 1))
    alpha = 16 * (data[_p] - DELAY(data[_p], 1) + (data[_p] - data["open"]) / 2 +
                  DELAY(data[_p], 1) - DELAY(data["open"], 1))
    alpha = alpha / (IFELSE((abshc > abslc) * (abshc > abshl), abshc + abslc / 2 + absco / 4,
                            IFELSE((abslc > abshl) * (abslc > abshc),
                                   abslc + abshc / 2 + absco / 4, abshl + absco / 4)) + 1e-7)
    alpha = alpha * MAX(abshc, abslc)
    return alpha


@factor_attr(max_depend=200, return_type="float")
def alpha191_138(data):
    """
    (
        RANK(DECAYLINEAR(DELTA((((LOW * 0.7) + (VWAP * 0.3))), 3), 20)) -
        TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(LOW, 8), TSRANK(MEAN(VOLUME, 60), 17), 5), 19), 16), 7)
    ) * -1
    @res:
    """
    alpha = (RANK(DECAYLINEAR(DELTA((((data["low"] * 0.7) + (data["vwap"] * 0.3))), 3), 20)) -
             TS_RANK(
                 DECAYLINEAR(TS_RANK(CORR(TS_RANK(data["low"], 8), TS_RANK(MEAN(data["volume"], 60), 17), 5), 19), 16),
                 7)) * -1
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_139(data, period=10):
    """
    (-1 * CORR(OPEN, VOLUME, 10))
    @res:
    """
    alpha = (-1 * CORR(data["open"], data["volume"], period))
    return alpha


@factor_attr(max_depend=150, return_type="float")
def alpha191_140(data):
    """
    MIN(
        RANK(DECAYLINEAR(((RANK(OPEN) + RANK(LOW)) - (RANK(HIGH) + RANK(CLOSE))), 8)),
        TSRANK(DECAYLINEAR(CORR(TSRANK(CLOSE, 8), TSRANK(MEAN(VOLUME, 60), 20), 8), 7), 3)
    )
    @res:
    """
    alpha = np.minimum(
        RANK(DECAYLINEAR(((RANK(data["open"]) + RANK(data["low"])) - (RANK(data["high"]) + RANK(data["close"]))), 8)),
        TS_RANK(DECAYLINEAR(CORR(TS_RANK(data["close"], 8), TS_RANK(MEAN(data["volume"], 60), 20), 8), 7), 3)
    )
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_141(data):
    """
    (RANK(CORR(RANK(HIGH), RANK(MEAN(VOLUME,15)), 9))* -1)
    @res:
    """
    alpha = (RANK(CORR(RANK(data["high"]), RANK(MEAN(data["volume"], 15)), 9)) * -1)
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_142(data):
    """
    (((-1 * RANK(TSRANK(CLOSE, 10))) * RANK(DELTA(DELTA(CLOSE, 1), 1))) * RANK(TSRANK((VOLUME
    /MEAN(VOLUME,20)), 5)))
    @res:
    """
    alpha = (((-1 * RANK(TS_RANK(data["close"], 10))) * RANK(DELTA(DELTA(data["close"], 1), 1))) *
             RANK(TS_RANK((data["volume"] / MEAN(data["volume"], 20)), 5)))
    return alpha


@factor_attr(unfinished=True)
def alpha191_143(data, use_vwap=False):
    """
    CLOSE>DELAY(CLOSE,1)?(CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*SELF:SELF
    @res:
    """
    # _p = "vwap" if use_vwap else "close"
    # delay = DELAY(data[_p], 1)
    # alpha = pd.DataFrame(_alpha191_143(data[_p].values, delay.values), index=delay.index, columns=delay.columns)
    # return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_144(data):
    """
    SUMIF(ABS(CLOSE / DELAY(CLOSE, 1) - 1) / AMOUNT, 20, CLOSE < DELAY(CLOSE, 1)) /
    COUNT(CLOSE < DELAY(CLOSE, 1), 20)
    @res:
    """
    part = np.abs(data["close"] / DELAY(data["close"], 1) - 1) / np.log(data["amount"])
    alpha = IFELSE(data["close"] < DELAY(data["close"], 1), part, 0)
    na_map = alpha.isna().astype(float).replace(1., np.nan)
    alpha = SUM(alpha, 20) / COUNT(data["close"] < DELAY(data["close"], 1), 20, na_map=na_map)
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_145(data):
    """
    (MEAN(VOLUME,9)-MEAN(VOLUME,26))/MEAN(VOLUME,12)*100
    @res:
    """
    alpha = (MEAN(data["volume"], 9) - MEAN(data["volume"], 26)) / MEAN(data["volume"], 12) * 100
    return alpha


@factor_attr(max_depend=100, return_type="float")
def alpha191_146(data, period=61, use_vwap=False):
    """
    part = (CLOSE - DELAY(CLOSE, 1)) / DELAY(CLOSE, 1)
    part_ma = part - SMA(part, 61, 2)
    MEAN(part1, 20) * (part_ma) / SMA((part - (part_ma)) ^ 2, 60)
    @res:
    """
    _p = "vwap" if use_vwap else "close"
    part = (data[_p] - DELAY(data[_p], 1)) / DELAY(data[_p], 1)
    part_ma = part - SMA(part, period, 2)
    alpha = MEAN(part_ma, 20) * (part_ma) / SMA((part - (part_ma)) ** 2, period, 2)
    return alpha


@factor_attr(max_depend=50, return_type="binary")
def alpha191_147(data, period=12, use_vwap=False):
    """
    REGBETA(MEAN(CLOSE,12),SEQUENCE(12))
    @res:
    """
    _p = "vwap" if use_vwap else "close"
    idx = data[_p].index
    alpha = MEAN(data[_p], period).apply(lambda x: pd.Series(rolling_slope(x.values, period)))
    alpha.index = idx
    return alpha


@factor_attr(max_depend=100, return_type="binary")
def alpha191_148(data):
    """
    (RANK(CORR((OPEN), SUM(MEAN(VOLUME, 60), 9), 6)) < RANK((OPEN - TSMIN(OPEN, 14)))) * -1
    @res:
    """
    alpha = RANK(CORR((data["open"]), SUM(MEAN(data["volume"], 60), 9), 6))
    na_map = alpha.isna().astype(float).replace(1., np.nan)
    alpha = (alpha < RANK((data["open"] - TS_MIN(data["open"], 14)))).replace(False, -1) * -1
    return alpha + na_map


@factor_attr(max_depend=300, return_type="float")
def alpha191_149(data, period=252):
    """
    REGBETA(
        FILTER(CLOSE / DELAY(CLOSE, 1) - 1, BANCHMARKINDEXCLOSE < DELAY(BANCHMARKINDEXCLOSE, 1)),
        FILTER(BANCHMARKINDEXCLOSE / DELAY(BANCHMARKINDEXCLOSE, 1) - 1, BANCHMARKINDEXCLOSE < DELAY(BANCHMARKINDEXCLOSE, 1)),
    252)
    @res:
    """
    bench = (data["close"] / DELAY(data["close"], 1) - 1).mean(axis=1)
    # condition = bench < DELAY(bench, 1)
    alpha = REGBETA(data["close"] / DELAY(data["close"], 1) - 1, bench, period)
    # alpha = alpha[condition]
    return alpha


@factor_attr(max_depend=10, return_type="float")
def alpha191_150(data):
    """
    (CLOSE+HIGH+LOW)/3*VOLUME
    @res:
    """
    alpha = (data["close"] + data["high"] + data["low"]) / 3 * np.log(data["volume"])
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_151(data, period=20, use_vwap=False):
    """
    SMA(CLOSE-DELAY(CLOSE,20),20,1)
    @res:
    """
    _p = "vwap" if use_vwap else "close"
    alpha = SMA(data[_p] - DELAY(data[_p], period), period, 1)
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_152(data, short_period=12, long_period=26, use_vwap=False):
    """
    SMA(MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),12)-MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),26),9,1)
    @res:
    """
    _p = "vwap" if use_vwap else "close"
    part = DELAY(SMA(DELAY(data[_p] / DELAY(data[_p], 9), 1), 9, 1), 1)
    alpha = SMA(MEAN(part, short_period) - MEAN(part, long_period), 9, 1)
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_153(data, use_vwap=False):
    """
    (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/4
    @res:
    """
    _p = "vwap" if use_vwap else "close"
    alpha = (MEAN(data[_p], 3) + MEAN(data[_p], 6) + MEAN(data[_p], 12) + MEAN(data[_p], 24)) / 4
    return alpha


@factor_attr(max_depend=300, return_type="binary")
def alpha191_154(data):
    """
    ((VWAP - MIN(VWAP, 16))) < (CORR(VWAP, MEAN(VOLUME,180), 18))
    @res:
    """
    alpha = ((data["vwap"] - TS_MIN(data["vwap"], 16)))  # < (CORR(data["vwap"], MEAN(data["volume"], 180), 18))
    na_map = alpha.isna().astype(float).replace(1., np.nan)
    alpha = (alpha < CORR(data["vwap"], MEAN(data["volume"], 180), 18)).replace(False, -1)
    return alpha + na_map


@factor_attr(max_depend=50, return_type="float")
def alpha191_155(data):
    """
    SMA(VOLUME,13,2)-SMA(VOLUME,27,2)-SMA(SMA(VOLUME,13,2)-SMA(VOLUME,27,2),10,2)
    @res:
    """
    alpha = SMA(data["volume"], 13, 2) - SMA(data["volume"], 27, 2) - SMA(
        SMA(data["volume"], 13, 2) - SMA(data["volume"], 27, 2), 10, 2)
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_156(data, period=3):
    """
    MAX(
        RANK(DECAYLINEAR(DELTA(VWAP, 5), 3)),
        RANK(DECAYLINEAR(((
            DELTA(((OPEN * 0.15) + (LOW * 0.85)), 2) /
            ((OPEN * 0.15) + (LOW * 0.85))) * -1), 3))
    ) * -1
    @res:
    """
    alpha = RANK(DECAYLINEAR((data["vwap"] - DELAY(data["vwap"], 5)), period))
    alpha = np.maximum(alpha, RANK(DECAYLINEAR(((DELTA(((data["open"] * 0.15) + (data["low"] * 0.85)), 2) /
                                                 ((data["open"] * 0.15) + (data["low"] * 0.85))) * -1), period)))
    alpha = alpha * -1
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_157(data):
    """
    MIN(RANK(RANK(LOG(SUM(TSMIN(RANK(RANK((-1 * RANK(DELTA((CLOSE - 1), 5))))), 2), 1)))), 5) +
    TSRANK(DELAY((-1 * RET), 6), 5)
    @res:
    """
    alpha = TS_MIN(RANK(RANK(np.log(SUM(TS_MIN(RANK(RANK((-1 * RANK(DELTA((data["close"] - 1), 5))))), 2), 1)))), 5) + \
            TS_RANK(DELAY((-1 * data["close"].pct_change()), 6), 5)
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_158(data, period=15, use_vwap=False):
    """
    ((HIGH - SMA(CLOSE, 15, 2)) - (LOW - SMA(CLOSE, 15, 2))) / CLOSE
    @res:
    """
    _p = "vwap" if use_vwap else "close"
    alpha = ((data["high"] - SMA(data[_p], period, 2)) - (data["low"] - SMA(data[_p], period, 2))) / \
            data[_p]
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_159(data):
    """
    part1 = part3 - part2;
    part2 = MIN(LOW, DELAY(CLOSE, 1));
    part3 = MAX(HGIH, DELAY(CLOSE, 1));
    (
        (CLOSE - SUM(part2, 6)) /
        SUM(part1, 6) * 288 +
        (CLOSE - SUM(part2, 12)) / SUM(part3 - part2, 12) * 144 +
        (CLOSE - SUM(part2, 24)) /
        SUM(part1, 24) * 144)
    * 100 / 504
    @res:
    """
    part2 = np.minimum(data["low"], DELAY(data["close"], 1))
    part3 = np.maximum(data["high"], DELAY(data["close"], 1))
    part1 = part3 - part2
    alpha = (data["close"] - SUM(part2, 6)) / SUM(part1, 6) * 288 + \
            (data["close"] - SUM(part2, 12)) / SUM(part3 - part2, 12) * 144 + \
            (data["close"] - SUM(part2, 24)) / SUM(part1, 24) * 144
    alpha = alpha * 100 / 504
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_160(data, period=20, use_vwap=False):
    """
    SMA((CLOSE <= DELAY(CLOSE, 1) ? STD(CLOSE, 20) : 0), 20, 1)
    @res:
    """
    _p = "vwap" if use_vwap else "close"
    part = STD(data[_p], period)
    alpha = SMA(IFELSE(data[_p] <= DELAY(data[_p], 1), part, 0), period, 1)
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_161(data, period=12):
    """
    MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),12)
    @res:
    """
    alpha = MEAN(MAX(MAX((data["high"] - data["low"]), np.abs(DELAY(data["close"], 1) - data["high"])),
                     np.abs(DELAY(data["close"], 1) - data["low"])), period)
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_162(data, period=12, use_vwap=False):
    """
    prat1 = CLOSE - DELAY(CLOSE, 1);
    part2 = SMA(MAX(part1, 0), 12, 1);
    part3 = SMA(ABS(part1), 12, 1);
    part4 = MIN(part2 / part3 * 100, 12);
    (part2 / part3 * 100 - part4) / (MAX(part2 / part3 * 100, 12) - part4)
    @res:
    """
    _p = "vwap" if use_vwap else "close"
    part1 = data[_p] - DELAY(data[_p], 1)
    part2 = SMA(MAX(part1, 0), period, 1)
    part3 = SMA(np.abs(part1), period, 1)
    part4 = TS_MIN(part2 / part3 * 100, period)
    alpha = (part2 / part3 * 100 - part4) / (TS_MAX(part2 / part3 * 100, period) - part4)
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_163(data, period=20):
    """
    RANK(((((-1 * RET) * MEAN(VOLUME,20)) * VWAP) * (HIGH - CLOSE)))
    @res:
    """
    ret = data["close"] / data["close"].shift(1) - 1
    alpha = RANK(((((-1 * ret) * MEAN(data["volume"], period)) * data["vwap"]) * (data["high"] - data["close"])))
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_164(data, use_vwap=False):
    """
    SMA((
            ((CLOSE > DELAY(CLOSE, 1)) ? 1 / (CLOSE - DELAY(CLOSE, 1)) : 1) -
            MIN(((CLOSE > DELAY(CLOSE, 1)) ? 1 / (CLOSE - DELAY(CLOSE, 1)) : 1), 12)
        ) /
        (HIGH - LOW) * 100, 13, 2)
    -->
    diff = (CLOSE - DELAY(CLOSE, 1));
    conditon = (CLOSE > DELAY(CLOSE, 1));
    SMA((
            (conditiopn ? 1 / diff : 1) -
            MIN((conditiopn ? 1 / diff : 1), 12)
        ) /
        (HIGH - LOW) * 100, 13, 2)
    @res:
    """
    _p = "vwap" if use_vwap else "close"
    diff = data[_p] - DELAY(data[_p], 1)
    condition = (data[_p] > DELAY(data[_p], 1))
    alpha = SMA(IFELSE(condition, 1 / diff, 1) - TS_MIN(IFELSE(condition, 1 / diff, 1), 12) / (
            data["high"] - data["low"]) * 100, 13, 2)
    return alpha


@factor_attr(max_depend=200, return_type="float")
def alpha191_165(data, period=48, use_vwap=False):
    """
    MAX(SUMAC(CLOSE-MEAN(CLOSE,48)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,48)))/STD(CLOSE,48)
    -->
    MAX(SUMAC(diff)) - MIN(SUMAC(diff)) / STD(CLOSE, 48)
    @res:
    """
    _p = "vwap" if use_vwap else "close"
    diff = data[_p] - MEAN(data[_p], period)
    alpha = TS_MAX(SUM(diff, 48), 48) - TS_MIN(SUM(diff, 48), 48) / STD(data[_p], period)
    return alpha


@factor_attr(max_depend=100, return_type="float")
def alpha191_166(data):
    """
    -20 * (19) ^ 1.5 * SUM(CLOSE / DELAY(CLOSE, 1) - 1 - MEAN(CLOSE / DELAY(CLOSE, 1) - 1, 20), 20) /
    ((20 - 1) * (20 - 2)(SUM((CLOSE / DELAY(CLOSE, 1), 20) ^ 2, 20)) ^ 1.5)
    ->
    5 * SUM(CLOSE / DELAY(CLOSE, 1) - 1 - MEAN(CLOSE / DELAY(CLOSE, 1) - 1, 20), 20) /
    (SUM(MEAN(CLOSE / DELAY(CLOSE, 1), 20) ^ 2, 20)) ^ 1.5
    @res:
    """
    part = data["close"] / DELAY(data["close"], 1)
    alpha = 5 * SUM(part - 1 - MEAN(part - 1, 20), 20) / (SUM(MEAN(part, 20) ** 2, 20)) ** 1.5
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_167(data, period=12, use_vwap=False):
    """
    SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)
    @res:
    """
    _p = "vwap" if use_vwap else "close"
    part = data[_p] - DELAY(data[_p], 1)
    alpha = SUM(IFELSE(part > 0, part, 0), period)
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_168(data, period=20):
    """
    (-1*VOLUME/MEAN(VOLUME,20))
    @res:
    """
    alpha = (-1 * data["volume"] / MEAN(data["volume"], period))
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_169(data, use_vwap=False):
    """
    SMA(
        MEAN(DELAY(SMA(CLOSE - DELAY(CLOSE, 1), 9, 1), 1), 12) -
        MEAN(DELAY(SMA(CLOSE - DELAY(CLOSE, 1), 9, 1), 1), 26), 10, 1)
    @res:
    """
    _p = "vwap" if use_vwap else "close"
    part = SMA(data[_p] - DELAY(data[_p], 1), 9, 1)
    alpha = SMA(MEAN(DELAY(part, 1), 12) - MEAN(DELAY(part, 1), 26), 10, 1)
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_170(data):
    """
    (
        ((RANK((1 / CLOSE)) * VOLUME) / MEAN(VOLUME, 20)) *
        ((HIGH * RANK((HIGH - CLOSE))) / (SUM(HIGH, 5) / 5))
    ) -
    RANK((VWAP - DELAY(VWAP, 5)))
    @res:
    """
    alpha = RANK(1 / data["close"]) * data["volume"] / MEAN(data["volume"], 20)
    alpha *= (data["high"] * RANK(data["high"] - data["close"])) / (SUM(data["high"], 5) / 5)
    alpha -= RANK(data["vwap"] - DELAY(data["vwap"], 5))
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_171(data):
    """
    ((-1 * ((LOW - CLOSE) * (OPEN^5))) / ((CLOSE - HIGH) * (CLOSE^5)))
    @res:
    """
    alpha = (-1 * ((data["low"] - data["close"]) * (data["open"] ** 5))) / \
            ((data["close"] - data["high"] + 1e-7) * (data["close"] ** 5))
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_172(data):
    """
    MEAN(
	ABS(SUM((LD > 0 & LD > HD) ? LD : 0, 14) * 100 / SUM(TR, 14) - SUM((HD > 0 & HD > LD) ? HD : 0, 14) * 100 / SUM(TR, 14)) /
	(SUM((LD > 0 & LD > HD) ? LD : 0, 14) * 100 / SUM(TR, 14) + SUM((HD > 0 & HD > LD) ? HD : 0, 14) * 100 / SUM(TR, 14)) * 100, 6)
	# TR = MAX(MAX(HIGH - LOW, ABS(HIGH - DELAY(CLOSE, 1))), ABS(LOW - DELAY(CLOSE, 1)))
    # HD = HIGH-DELAY(HIGH,1)
    # LD = DELAY(LOW,1)-LOW
    @res:
    """
    tr = MAX(MAX(data["high"] - data["low"], np.abs(data["high"] - DELAY(data["close"], 1))),
             np.abs(data["low"] - DELAY(data["close"], 1)))
    hd = data["high"] - DELAY(data["high"], 1)
    ld = DELAY(data["low"], 1) - data["low"]
    part1 = SUM(IFELSE((ld > 0) & (ld > hd), ld, 0), 14) * 100 / SUM(tr, 14)
    part2 = SUM(IFELSE((hd > 0) & (hd > ld), hd, 0), 14) * 100 / SUM(tr, 14)
    alpha = MEAN(np.abs(part1 - part2) / (part1 + part2) * 100, 6)
    return alpha


@factor_attr(max_depend=100, return_type="float")
def alpha191_173(data, period=13, use_vwap=False):
    """
    3 * SMA(CLOSE, 13, 2) - 2 * SMA(SMA(CLOSE, 13, 2), 13, 2) + SMA(SMA(SMA(LOG(CLOSE), 13, 2), 13, 2), 13, 2);
    @res:
    """
    _p = "vwap" if use_vwap else "close"
    ma = SMA(data[_p], period, 2)
    alpha = 3 * ma - 2 * SMA(ma, period, 2) + SMA(SMA(np.log(data[_p]), period, 2), period, 2)
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_174(data, period=20, use_vwap=False):
    """
    SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)
    @res:
    """
    _p = "vwap" if use_vwap else "close"
    part = IFELSE(data[_p] > DELAY(data[_p], 1), STD(data[_p], period), 0)
    alpha = SMA(part, period, 1)
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_175(data, period=6):
    """
    MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),6)
    @res:
    """
    alpha = MEAN(MAX(MAX((data["high"] - data["low"]), np.abs(DELAY(data["close"], 1) - data["high"])),
                     np.abs(DELAY(data["close"], 1) - data["low"])), period)
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_176(data, period=12, corr_period=6):
    """
    CORR(RANK(((CLOSE - TSMIN(LOW, 12)) / (TSMAX(HIGH, 12) - TSMIN(LOW,12)))), RANK(VOLUME), 6)
    @res:
    """
    alpha = CORR(RANK((data["close"] - TS_MIN(data["low"], period)) /
                      (TS_MAX(data["high"], period) - TS_MIN(data["low"], period))),
                 RANK(data["volume"]), corr_period)
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_177(data, period=20):
    """
    ((20-HIGHDAY(HIGH,20))/20)*100
    @res:
    """
    alpha = (20 - HIGHDAY(data["high"], period)) / 20 * 100
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_178(data):
    """
    (CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*VOLUME
    @res:
    """
    alpha = (data["close"] - DELAY(data["close"], 1)) / DELAY(data["close"], 1) * data["volume"]
    return alpha


@factor_attr(max_depend=100, return_type="float")
def alpha191_179(data):
    """
    (RANK(CORR(VWAP, VOLUME, 4)) *RANK(CORR(RANK(LOW), RANK(MEAN(VOLUME,50)), 12)))
    @res:
    """
    alpha = RANK(CORR(data["vwap"], data["volume"], 4)) * \
            RANK(CORR(RANK(data["low"]), RANK(MEAN(data["volume"], 50)), 12))
    return alpha


@factor_attr(max_depend=100, return_type="float")
def alpha191_180(data):
    """
    MEAN(VOLUME,20) < VOLUME ?
        (-1 * TSRANK(ABS(DELTA(CLOSE, 7)), 60)) * SIGN(DELTA(CLOSE, 7)) :
        -VOLUME
    @res:
    """
    alpha = IFELSE(MEAN(data["volume"], 20) < data["volume"],
                   (-1 * TS_RANK(np.abs(DELTA(data["close"], 7)), 60)) * np.sign(DELTA(data["close"], 7)),
                   - data["volume"])
    return alpha


@factor_attr(max_depend=100, return_type="float")
def alpha191_181(data):
    """
    SUM(
        ((CLOSE / DELAY(CLOSE, 1) - 1) - MEAN((CLOSE / DELAY(CLOSE, 1) - 1), 20)) -
        (BANCHMARKINDEXCLOSE - MEAN(BANCHMARKINDEXCLOSE, 20)) ^ 2, 20) /
    SUM((BANCHMARKINDEXCLOSE - MEAN(BANCHMARKINDEXCLOSE, 20)) ^ 3)
    @res:
    """
    bench = (data["close"] / DELAY(data["close"], 1) - 1).mean(axis=1)
    ret = (data["close"] / DELAY(data["close"], 1) - 1)
    part = bench - bench.rolling(20).mean()
    alpha = ret - MEAN(ret, 20)
    alpha = SUM(alpha.sub(part ** 2, axis="index"), 20).div(SUM(part ** 3, 20), axis="index")
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_182(data, period=20):
    """
    COUNT(
        (CLOSE > OPEN & BANCHMARKINDEXCLOSE > BANCHMARKINDEXOPEN) |
        (CLOSE < OPEN & BANCHMARKINDEXCLOSE < BANCHMARKINDEXOPEN), 20) / 20
	@res:
    """
    bench = data["close"].mean(axis=1) > data["open"].mean(axis=1)
    part = data["close"] > data["open"]
    na_map = data["close"].isna().astype(float).replace(1., np.nan)
    alpha = part.eq(bench, axis="index") + na_map
    alpha = SUM(alpha, period) / period
    return alpha


@factor_attr(max_depend=100, return_type="float")
def alpha191_183(data, period=24, use_vwap=False):
    """
    MAX(SUMAC(CLOSE-MEAN(CLOSE,24)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,24)))/STD(CLOSE,24)
    @res:
    """
    _p = "vwap" if use_vwap else "close"
    part = SUM(data[_p] - MEAN(data[_p], period), period)
    alpha = TS_MAX(part, period) - TS_MIN(part, period) / STD(data[_p], period)
    return alpha


@factor_attr(max_depend=300, return_type="float")
def alpha191_184(data):
    """
    (RANK(CORR(DELAY((OPEN - CLOSE), 1), CLOSE, 200)) + RANK((OPEN - CLOSE)))
    @res:
    """
    alpha = RANK(CORR(DELAY((data["open"] - data["close"]), 1), data["close"], 200)) + \
            RANK((data["open"] - data["close"]))
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_185(data):
    """
    RANK((-1 * ((1 - (OPEN / CLOSE))^2)))
    @res:
    """
    alpha = RANK((-1 * ((1 - (data["open"] / data["close"])) ** 2)))
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_186(data, period=14, ma_period=6):
    """
    part1 = SUM((LD > 0 & LD > HD) ? LD : 0, 14) * 100 / SUM(TR, 14);
    part2 = SUM((HD > 0 & HD > LD) ? HD : 0, 14) * 100 / SUM(TR, 14);
    part3 = ABS(part1 - part2) / (part1 + part2) * 100;
    (MEAN(part3, 6) + DELAY(MEAN(part3, 6), 6)) / 2
    @res:
    """
    tr = MAX(MAX(data["high"] - data["low"], np.abs(data["high"] - DELAY(data["close"], 1))),
             np.abs(data["low"] - DELAY(data["close"], 1)))
    hd = data["high"] - DELAY(data["high"], 1)
    ld = DELAY(data["low"], 1) - data["low"]
    sum_tr = SUM(tr, period)
    part1 = SUM(IFELSE((ld > 0) & (ld > hd), ld, 0), period) * 100 / sum_tr
    part2 = SUM(IFELSE((hd > 0) & (hd > ld), hd, 0), period) * 100 / sum_tr
    part3 = np.abs(part1 - part2) / (part1 + part2) * 100
    alpha = (MEAN(part3, ma_period) + DELAY(MEAN(part3, ma_period), ma_period)) / 2
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_187(data, period=20):
    """
    SUM((OPEN <= DELAY(OPEN, 1) ? 0 : MAX((HIGH - OPEN), (OPEN - DELAY(OPEN, 1)))), 20)
    @res:
    """
    alpha = SUM(IFELSE(data["open"] <= DELAY(data["open"], 1), 0,
                       MAX((data["high"] - data["open"]), (data["open"] - DELAY(data["open"], 1)))), period)
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_188(data, period=11):
    """
    ((HIGH-LOW–SMA(HIGH-LOW,11,2))/SMA(HIGH-LOW,11,2))*100
    @res:
    """
    alpha = ((data["high"] - data["low"] - SMA(data["high"] - data["low"], period, 2)) /
             SMA(data["high"] - data["low"], period, 2)) * 100
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_189(data, period=6, use_vwap=False):
    """
    MEAN(ABS(CLOSE-MEAN(CLOSE,6)),6)
    @res:
    """
    _p = "vwap" if use_vwap else "close"
    alpha = MEAN(np.abs(data[_p] - MEAN(data[_p], period)), period)
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_190(data, use_vwap=False):
    """
    part1 = CLOSE / DELAY(CLOSE) - 1;
    part2 = (CLOSE / DELAY(CLOSE, 19)) ^ (1 / 20) - 1;
    LOG(
        (COUNT(part1 > part2, 20) - 1) *
        (SUMIF(((part1 - part2)) ^ 2, 20, part1 < part2)) /
        (
            (COUNT((part1 < part2), 20)) *
            (SUMIF((part1 - part2) ^ 2, 20, part1 > part2))
        )
    )
    @res:
    """
    _p = "vwap" if use_vwap else "close"
    part1 = data[_p] / DELAY(data[_p], 1) - 1
    part2 = (data[_p] / DELAY(data[_p], 19)) ** (1 / 20) - 1
    part3 = ((part1 - part2) ** 2)
    na_map = part3.isna().astype(float).replace(1., np.nan)
    alpha = np.log(
        (COUNT(part1 > part2, 20, na_map=na_map) - 1) *
        (SUM(IFELSE(part1 < part2, part3, 0), 20)) /
        (
                (COUNT((part1 < part2), 20, na_map=na_map)) *
                (SUM(IFELSE(part1 > part2, part3, 0), 20))
        )
    )
    return alpha


@factor_attr(max_depend=50, return_type="float")
def alpha191_191(data):
    """
    ((CORR(MEAN(VOLUME,20), LOW, 5) + ((HIGH + LOW) / 2)) - CLOSE)
    @res:
    """
    alpha = ((CORR(MEAN(data["volume"], 20), data["low"], 5) + ((data["high"] + data["low"]) / 2)) - data["close"])
    return alpha


# FACTOR_ALPHA191 = {}
# for i in range(1, 192):
#     _func = getattr(sys.modules[__name__], "alpha191_%03d" % i)
#     if not _func.unfinished:
#         FACTOR_ALPHA191["alpha191_%03d" % i] = _func


def _get_alpha191_fn(start=1, end=191):
    """
    获取alpha191因子函数
    :param start: 起始因子
    :param end: 结束因子
    :return: 因子函数列表
    """
    alpha191_funcs = {}
    for i in range(start, end + 1):
        _func = getattr(sys.modules[__name__], "alpha191_%03d" % i)
        if not _func.unfinished:
            alpha191_funcs["alpha191_%03d" % i] = _func
    return alpha191_funcs


class Alpha191(FactorBase):
    def __init__(self, name, desc):
        super(Alpha191, self).__init__(name, desc)

    def _check_input(self, data: Dict[str, pd.DataFrame]):
        _indx = None
        _col = None
        for field in data.keys():
            if _indx is None:
                _indx = data[field].index
                _col = data[field].columns
            else:
                _indx1 = data[field].index
                _col1 = data[field].columns
                if (_indx == _indx1).all() and (_col == _col1).all():
                    print(f"_check_input pass {field}")
                else:
                    raise ValueError(f"index or columns not match {field}")

    def _prepare_data(self, input_data: Dict[str, pd.DataFrame]):
        fileds = list(input_data.keys())
        columns = input_data[fileds[0]].columns
        shape = input_data[fileds[0]].shape
        assert "date" not in columns, "date is in columns, the input data must be indexed by date"
        assert shape[0] > 300 + self.diff_days, f"the input data must have more than {300 + self.diff_days} rows"
        self.data = {}
        for field in fileds:
            assert input_data[field].shape == shape, f"{field} shape is not equal to {fileds[0]}"
            self.data[field] = input_data[field].reindex(columns=columns).copy()

        self.data["vwap"] = self.data["amount"] / self.data["volume"]
        self.data["turn"] = self.data["amount"] / self.data["liquidity_value"]
        self._check_input(self.data)

    def _split_data(self):
        group = [g + self.diff_days for g in [50, 100, 150, 200, 300]]
        data = []
        for i in range(len(group)):
            data.append({field: self.data[field].iloc[-group[i]:] for field in self.data.keys()})
        self.data = data

    def _map_depend(self, max_depend):
        # for g in [50, 100, 150, 200]:
        if max_depend < 50:
            return self.data[0]
        elif max_depend < 100:
            return self.data[1]
        elif max_depend < 150:
            return self.data[2]
        elif max_depend < 200:
            return self.data[3]
        else:
            return self.data[4]

    def init(self,
             input_data: Dict[str, pd.DataFrame],
             diff_days: int,
             by_depend=False,
             id_start: int = 1,
             id_end: int = 191,
             verbose=True):
        """
        :param input_data: Dict[str, pd.DataFrame]
        :param diff_days: int 需计算的时间跨度
        :param by_depend: bool, default False, 是否基于依赖长度计算因子(减少冗余计算, 当结果只取末尾几天时, 可以设置为True)
        :param id_start: int, default 1, 起始因子编号
        :param id_end: int, default 191, 结束因子编号
        :param verbose: bool, default False, 是否使用tqdm打印因子计算进度
        """
        self.diff_days = diff_days
        self.by_depend = by_depend
        self._prepare_data(input_data)
        if by_depend:
            self._split_data()
        self.alpha191_funcs = _get_alpha191_fn(id_start, id_end)
        self.verbose = verbose

    def cal(self, **kwargs) -> Dict[str, pd.DataFrame]:
        import os
        res = {}
        if not self.by_depend:
            for fac_name, fac_func in self.alpha191_funcs.items():
                if not fac_func.unfinished:
                    if self.verbose:
                        print(f"cal {fac_name}, pid: {os.getpid()}")
                    try:
                        res[fac_name] = fac_func(self.data)
                    except Exception as e:
                        print(f"{fac_name} error {e}")
        else:
            self._split_data()
            for fac_name, fac_func in self.alpha191_funcs.items():
                if not fac_func.unfinished:
                    if self.verbose:
                        print(f"cal {fac_name}, pid: {os.getpid()}")
                    try:
                        res[fac_name] = fac_func(self._map_depend(fac_func.max_depend))
                    except Exception as e:
                        print(f"{fac_name} error {e}")
        return res


"""
alpha 191 max depend count
{10: 18,
 30: 40,
 20: 4,
 300: 9,
 40: 3,
 50: 87,
 100: 21,
 60: 1,
 200: 5,
 150: 1}
"""