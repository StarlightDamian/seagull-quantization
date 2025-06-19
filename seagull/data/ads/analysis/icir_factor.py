# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 14:33:33 2025

@author: awei

icir_factor
"""
import os

import pandas as pd
import numpy as np
from joblib import Parallel, delayed

from seagull.settings import PATH
from analysis import icir
from seagull.utils import utils_log, utils_thread, utils_data, utils_database, utils_math

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')

#class FactorAnalyzer:
#    def __init__(self):
    
def calculate_close_rate(df: pd.DataFrame) -> pd.DataFrame:
    # df = df.sort_values(by='date', ascending=True)
    df[['close_rate']] = df[['close_rate']].shift(-1)
    
    # 1. 去极值
    #df[numeric_features] = winsorize(df[numeric_features])
    
    # 2. 中性化
    #neutralize
    
    # 3. 标准化
def calculate_close_rate(df: pd.DataFrame) -> pd.DataFrame:
    # df = df.sort_values(by='date', ascending=True)
    df[['close_rate']] = df[['close_rate']].shift(-1)
    
    #df[numeric_features] = df[numeric_features].apply(utils_math.log_e)
    #df[numeric_features] = winsorize(df[numeric_features])
    #df[numeric_features] = standardize(df[numeric_features])
    return df

def standardize(data):
    """
    标准化处理
    """
    return (data - data.mean()) / data.std()

def neutralize(factor_data, market_cap, industry_dummies):
    """
    市值、行业中性化
    """
    import statsmodels.api as sm

    # 准备回归变量
    X = pd.concat([np.log(market_cap), industry_dummies], axis=1)
    X = sm.add_constant(X)
    
    # 对每个时间点进行横截面回归
    residuals = pd.DataFrame(index=factor_data.index, columns=factor_data.columns)
    
    for date in factor_data.columns:
        y = factor_data.loc[date]
        mask = ~(y.isna() | X.loc[date].isna().any(axis=1))
        if mask.sum() > 0:
            model = sm.OLS(y[mask], X.loc[date][mask], missing='drop')
            residuals.loc[date] = model.fit().resid
    
    return residuals

def winsorize(df, n=3):
    """
    极值处理，逐列进行裁剪（使用向量化操作）。
    """
    # 计算每列的均值和标准差
    means = df.mean(axis=0)
    stds = df.std(axis=0)

    # 计算上限和下限
    upper = means + n * stds
    lower = means - n * stds

    # 使用广播（vectorized operation）来进行裁剪
    return np.clip(df.values, lower.values, upper.values)

def process_feature(feature_name, factor_df, price_data):
    try:
        # 计算IC和IR
        ic, ir = icir.calculate_icir(factor_df[feature_name], price_data)
        logger.info(feature_name)
        logger.info(ic.describe())  # df.describe(percentiles=[.25, .5, .75, .70])
        logger.info(ir)
        
        # 处理IC的描述统计
        icir_df_1 = ic.describe().T
        icir_df_1.index = ['1d', '5d', '10d', '22d']
        icir_df_1.columns = ['date_num', 'mean_ic', 'std_ic', 'min_ic', '25pct_ic',
                             '50pct_ic', '75pct_ic', 'max_ic']
        icir_df_1['freq'] = icir_df_1.index
        icir_df_1['feature_name'] = feature_name
        icir_df_1['ir'] = ir.values
        icir_df_1['remark'] = 'loge' # 标准化,无操作,标准化&极值处理
        
        # 输出到数据库
        utils_data.output_database(icir_df_1,
                                   filename='ads_info_incr_icir',
                                   if_exists='append',
                                   )
        
    except Exception as e:
        logger.error(f'{feature_name} {e}')
        
def analyze_factors_parallel_joblib(features, factor_df, price_data, num_workers=8):
    # 使用multiprocessing后端
    Parallel(n_jobs=num_workers, backend='multiprocessing')(
        delayed(process_feature)(feature_name, factor_df, price_data) for feature_name in features
    )


if __name__ == "__main__":
    raw_df = pd.read_feather(f'{PATH}/_file/das_wide_incr_train.feather')

    raw_df['date'] = pd.to_datetime(raw_df['date']).dt.tz_localize('UTC')
    alpha_features = ['alpha001', 'alpha002', 'alpha003', 'alpha004', 'alpha005', 'alpha006', 'alpha007', 'alpha008', 'alpha009', 'alpha010', 'alpha011', 'alpha012', 'alpha013', 'alpha014', 'alpha015', 'alpha016', 'alpha017', 'alpha018', 'alpha019', 'alpha020', 'alpha021', 'alpha022', 'alpha023', 'alpha024', 'alpha025', 'alpha026', 'alpha027', 'alpha028', 'alpha029', 'alpha030', 'alpha031', 'alpha032', 'alpha033', 'alpha034', 'alpha035', 'alpha036', 'alpha037', 'alpha038', 'alpha039', 'alpha040', 'alpha041', 'alpha042', 'alpha043', 'alpha044', 'alpha045', 'alpha046', 'alpha047', 'alpha049', 'alpha050', 'alpha051', 'alpha052', 'alpha053', 'alpha054', 'alpha055', 'alpha057', 'alpha060', 'alpha061', 'alpha062', 'alpha064', 'alpha065', 'alpha066', 'alpha068', 'alpha071', 'alpha072', 'alpha073', 'alpha074', 'alpha075', 'alpha077', 'alpha078', 'alpha081', 'alpha083', 'alpha084', 'alpha085', 'alpha086', 'alpha088', 'alpha092', 'alpha094', 'alpha095', 'alpha096', 'alpha098', 'alpha099', 'alpha101']

    features = alpha_features  # ohlc_features +\
               #fundamental_features +\
               #label_features +\
               #macd_features +\
               #alpha_features #+\
               #index_features +\
               #indicators_features
    #factor_df = raw_df[features]
    numeric_features = list(set(features)-
                            set(['alpha061','alpha062','alpha064','alpha065','alpha068','alpha074','alpha075',
                                 'alpha081','alpha086','alpha095','alpha099']))
    grouped = raw_df.groupby('full_code')
    raw_df = utils_thread.thread(grouped, calculate_close_rate, max_workers=8)
    
    price_data = raw_df.pivot_table(index='date', columns='full_code', values='close_rate')
    assets = price_data.loc[:, price_data.isna().sum()<3].columns  # sum(price_data.isna().sum()<3)==6082
    price_data = price_data[assets].ffill().bfill()
    
    factor_df = raw_df[raw_df.full_code.isin(assets)]
    factor_df = factor_df.set_index(['date', 'full_code'])
    analyze_factors_parallel_joblib(features, factor_df, price_data)

# =============================================================================
#     for feature_name in features: # = 'alpha041'
#         try:
#             ic, ir = icir3.icir(factor_df[feature_name], price_data)
#             logger.info(feature_name)
#             logger.info(ic.describe())  # df.describe(percentiles=[.25, .5, .75, .70])
#             logger.info(ir)
#             
#             icir_df_1 = ic.describe().T
#             icir_df_1.index = ['1d', '5d', '10d', '22d']
#             icir_df_1.columns = ['date_num', 'mean_ic', 'std_ic', 'min_ic', '25pct_ic',
#                                  '50pct_ic', '75pct_ic', 'max_ic']
#             icir_df_1['freq'] = icir_df_1.index
#             icir_df_1['feature_name'] = feature_name
#             icir_df_1['ir'] = ''
#             icir_df_1['remark'] = '标准化'
#             with utils_database.engine_conn("POSTGRES") as conn:
#                 utils_data.output_database(icir_df_1,
#                                            filename='ads_info_incr_icir')
#         except Exception as e:
#             logger.error(f'{feature_name} {e}')
# =============================================================================
    
    
    #df.alpha041 = standardize(df.alpha041)
# =============================================================================
#     raw_df.alpha002
#     Out[3]: 
#     0         -0.413994
#     1         -0.652837
#     2         -0.133620
#     3         -0.325072
#     4         -0.266600
#       
#     3195226   -0.230796
#     3195227   -0.107803
#     3195228   -0.434743
#     3195229   -0.724273
#     3195230   -0.097135
#     Name: alpha002, Length: 3195231, dtype: float64
# =============================================================================
# =============================================================================两个都有
# 2025-01-10 23:39:12.063 | INFO     | __main__:<module>:94 -                1D          5D         10D         22D
# count  465.000000  465.000000  465.000000  465.000000
# mean    -0.007820   -0.010537   -0.016816   -0.023011
# std      0.130768    0.130351    0.128999    0.130745
# min     -0.463308   -0.482947   -0.539532   -0.509202
# 25%     -0.075643   -0.074609   -0.091523   -0.100125
# 50%     -0.008988   -0.007697   -0.021589   -0.024724
# 75%      0.066419    0.058476    0.054221    0.045879
# max      0.539462    0.452066    0.412094    0.381407
# 2025-01-10 23:39:12.070 | INFO     | __main__:<module>:95 - 1D    -0.059803
# 5D    -0.080833
# 10D   -0.130359
# 22D   -0.176002
# dtype: float64
# =============================================================================
# =============================================================================两个都没
# 2025-01-10 23:41:06.205 | INFO     | __main__:<module>:94 -                1D          5D         10D         22D
# count  465.000000  465.000000  465.000000  465.000000
# mean    -0.003968   -0.000547   -0.001240   -0.002665
# std      0.121080    0.128935    0.128411    0.128154
# min     -0.386845   -0.352182   -0.352872   -0.349765
# 25%     -0.086400   -0.092781   -0.094818   -0.089760
# 50%     -0.007339    0.004839   -0.000902   -0.003990
# 75%      0.075548    0.087582    0.083121    0.085380
# max      0.333278    0.352784    0.313827    0.330399
# 2025-01-10 23:41:06.211 | INFO     | __main__:<module>:95 - 1D    -0.032775
# 5D    -0.004245
# 10D   -0.009655
# 22D   -0.020793
# dtype: float64
# =============================================================================
# =============================================================================只标准化
# 2025-01-10 23:45:19.776 | INFO     | __main__:<module>:94 -                1D          5D         10D         22D
# count  465.000000  465.000000  465.000000  465.000000
# mean    -0.007747   -0.010468   -0.016748   -0.022943
# std      0.130343    0.130023    0.128586    0.130305
# min     -0.462123   -0.480920   -0.536288   -0.509062
# 25%     -0.075559   -0.075397   -0.091006   -0.099098
# 50%     -0.008990   -0.006864   -0.021809   -0.023894
# 75%      0.067265    0.058940    0.054844    0.045322
# max      0.535671    0.449220    0.409788    0.379818
# 2025-01-10 23:45:19.784 | INFO     | __main__:<module>:95 - 1D    -0.059436
# 5D    -0.080510
# 10D   -0.130248
# 22D   -0.176068
# dtype: float64
# =============================================================================
