# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:31:12 2024

@author: awei
lightgbm_data

(lightgbm_data)
next_based_index_class

volume
"""
import os
import argparse

import pandas as pd
import numpy as np

from seagull.settings import PATH
from seagull.utils import utils_database, utils_log, utils_math, utils_data

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')

#def relative_price(df):
    # 这种操作通常被称为 去市场化（De-Marketization）或者 剔除市场趋势。
    
def _apply_relative_class(relative_close):
    # 根据条件分配 target_class
    if relative_close > 0.01:
        return 2
    elif -0.01 > relative_close:
        return 0
    #elif -0.05 <= relative_close <= 0.05:
    else:
        return 1

def _apply_rank_class(sub_df):
    sub_df['rank_rate'] = sub_df['next_relative_close_rel'].rank(pct=True)
    sub_df.loc[sub_df['rank_rate']>=0.66, 'target_class'] = 2
    sub_df.loc[(0.33<sub_df['rank_rate'])&(sub_df['rank_rate']<0.66), 'target_class'] = 1
    sub_df.loc[sub_df['rank_rate']<=0.33, 'target_class'] = 0
    return sub_df

def onehot(stock_df, label_df):
    # 对分类变量进行编码（如 One-Hot 编码）
    #df = pd.get_dummies(df, drop_first=True)
    # 或者使用 LabelEncoder 对分类标签进行编码（如有序标签）
    #from sklearn.preprocessing import LabelEncoder
    #le = LabelEncoder()
    #df['category_column'] = le.fit_transform(df['category_column'])

    # 示例数据
    #label_df = pd.DataFrame([['001', 'a'], ['001', 'b'], ['002', 'a']], columns=['full_code', 'label'])
    #stock_df = pd.DataFrame({'full_code': ['001', '002', '003']})
    
    # One-Hot 编码
    label_onehot = pd.get_dummies(label_df, columns=['label'], prefix='label_', prefix_sep='')

    # 聚合 One-Hot 编码
    label_onehot_grouped = label_onehot.groupby('full_code').max().reset_index()
    #del label_onehot_grouped['']
    
    # 将 One-Hot 编码结果与 stock_df 合并
    result_df = stock_df.merge(label_onehot_grouped, on='full_code', how='left').fillna(0)#.astype(int)

    #print(result_df)
    return result_df

def custom_loss(y_pred, y_true):
    #y_true = dataset.get_label()
    delta = y_true - y_pred
    alpha = 1.2  # Huber 损失的阈值参数，可以调整

    # 梯度和 Hessian
    grad = np.where(np.abs(delta) <= alpha, -delta, -alpha * np.sign(delta))
    hess = np.where(np.abs(delta) <= alpha, 0.9, 0.2)
    return grad, hess


def _apply_full_code(df):
    # df[['prev_close']] = df[['close']].shift(1)
    df[['high_rate', 'low_rate', 'open_rate', 'close_rate','avg_price_rate']] = df[['high', 'low', 'open', 'close','avg_price']].div(df['prev_close'], axis=0)
    df[['next_high_rate', 'next_low_rate', 'next_open_rate', 'next_close_rate', 'next_index_close_rate']] = df[['high_rate', 'low_rate', 'open_rate', 'close_rate','index_close_rate']].shift(-1)
    #df = df.head(-1)
    
    df['next_relative_close_rel'] = df['next_close_rate'] - df['next_index_close_rate']
    df['next_based_index_class'] = df['next_relative_close_rel'].apply(_apply_relative_class)
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='2023-01-03', help='When to start feature engineering')
    parser.add_argument('--date_end', type=str, default='2050-12-27', help='End time for feature engineering')
    args = parser.parse_args()
    
    logger.info(f"""task: train_table_merge
                    date_start: {args.date_start}
                    date_end: {args.date_end}""")
    
    # 获取日期段数据
    with utils_database.engine_conn("POSTGRES") as conn:
        df = pd.read_sql(f"SELECT * FROM dwd_ohlc_incr_stock_daily WHERE date BETWEEN '{args.date_start}' AND '{args.date_end}'", con=conn.engine)
        label_df = pd.read_sql("dwd_tags_full_label", con=conn.engine)
        macd_df = pd.read_sql(f"""
                    SELECT
                        primary_key
                        ,close_slope_12_26_9
                        ,volume_slope_12_26_9
                        ,turnover_slope_12_26_9
                        ,turnover_pct_slope_12_26_9
                        ,close_acceleration_12_26_9
                        ,volume_acceleration_12_26_9
                        ,turnover_acceleration_12_26_9
                        ,turnover_pct_acceleration_12_26_9
                        ,close_hist_12_26_9
                        ,volume_hist_12_26_9
                        ,turnover_hist_12_26_9
                        ,turnover_pct_hist_12_26_9
                        ,close_diff_1
                        ,close_diff_5
                        ,close_diff_30
                        ,volume_diff_1
                        ,volume_diff_5
                        ,volume_diff_30
                        ,turnover_diff_1
                        ,turnover_diff_5
                        ,turnover_diff_30
                        ,turnover_hist_diff_1
                        ,volume_hist_diff_1
                        ,close_hist_diff_1
                    FROM
                        dwd_feat_incr_macd
                    WHERE
                        date BETWEEN '{args.date_start}' AND '{args.date_end}'""", con=conn.engine)  # max_date=2024-08-14
# =============================================================================
#         flow_df = pd.read_sql(f"""
#                     SELECT
#                         primary_key
#                         ,loge_main_inflow
#                         ,loge_ultra_large_inflow
#                         ,loge_large_inflow
#                         ,loge_medium_inflow
#                         ,loge_small_inflow
#                         ,loge_main_small_net_inflow
#                         ,main_inflow_slope_12_26_9
#                         ,ultra_large_inflow_slope_12_26_9
#                         ,large_inflow_slope_12_26_9
#                         ,medium_inflow_slope_12_26_9
#                         ,small_inflow_slope_12_26_9
#                         ,main_small_net_inflow_slope_12_26_9
#                         ,main_inflow_acceleration_12_26_9
#                         ,ultra_large_inflow_acceleration_12_26_9
#                         ,large_inflow_acceleration_12_26_9
#                         ,medium_inflow_acceleration_12_26_9
#                         ,small_inflow_acceleration_12_26_9
#                         ,main_small_net_inflow_acceleration_12_26_9
#                         ,main_inflow_hist_12_26_9
#                         ,ultra_large_inflow_hist_12_26_9
#                         ,large_inflow_hist_12_26_9
#                         ,medium_inflow_hist_12_26_9
#                         ,small_inflow_hist_12_26_9
#                         ,main_small_net_inflow_hist_12_26_9
#                         ,main_inflow_diff_1
#                         ,main_inflow_diff_5
#                         ,main_inflow_diff_30
#                         ,ultra_large_inflow_diff_1
#                         ,ultra_large_inflow_diff_5
#                         ,ultra_large_inflow_diff_30
#                         ,large_inflow_diff_1
#                         ,large_inflow_diff_5
#                         ,large_inflow_diff_30
#                         ,medium_inflow_diff_1
#                         ,medium_inflow_diff_5
#                         ,medium_inflow_diff_30
#                         ,small_inflow_diff_1
#                         ,small_inflow_diff_5
#                         ,small_inflow_diff_30
#                         ,main_small_net_inflow_diff_1
#                         ,main_small_net_inflow_diff_5
#                         ,main_small_net_inflow_diff_30
#                         ,main_inflow_hist_diff_1
#                         ,ultra_large_inflow_hist_diff_1
#                         ,large_inflow_hist_diff_1
#                         ,medium_inflow_hist_diff_1
#                         ,small_inflow_hist_diff_1
#                         ,main_small_net_inflow_hist_diff_1
#                     FROM
#                         dwd_feat_incr_capital_flow
#                     WHERE
#                         date BETWEEN '{args.date_start}' AND '{args.date_end}'""", con=conn.engine)  # 2021-08-25, 2024-10-31
#                         
# =============================================================================
        alpha_df = pd.read_sql(f"""
                    SELECT
                        *
                    FROM
                        dwd_feat_incr_alpha
                    WHERE
                        date BETWEEN '{args.date_start}' AND '{args.date_end}'""", con=conn.engine)  # 2023-01-01, 2024-01-01
                        
        index_df = pd.read_sql(f"""
                    SELECT
                        --full_code
                        date
                        ,close as index_close 
                        ,volume as index_volume
                        ,turnover as index_turnover
                        ,turnover_pct as index_turnover_pct
                        --,primary_key
                        --,close_slope_12_26_9
                        --,volume_slope_12_26_9
                        --,turnover_slope_12_26_9
                        --,turnover_pct_slope_12_26_9
                        --,close_acceleration_12_26_9
                        --,volume_acceleration_12_26_9
                        --,turnover_acceleration_12_26_9
                        --,turnover_pct_acceleration_12_26_9
                        --,close_hist_12_26_9
                        --,volume_hist_12_26_9
                        --,turnover_hist_12_26_9
                        --,turnover_pct_hist_12_26_9
                        ,close_diff_1 as index_close_diff_1
                        ,close_diff_5 as index_close_diff_5
                        ,close_diff_30 as index_close_diff_30
                        ,volume_diff_1 as index_volume_diff_1
                        ,volume_diff_5 as index_volume_diff_5
                        ,volume_diff_30 as index_volume_diff_30
                        ,turnover_diff_1 as index_turnover_diff_1
                        ,turnover_diff_5 as index_turnover_diff_5
                        ,turnover_diff_30 as index_turnover_diff_30
                        --,turnover_hist_diff_1
                        --,volume_hist_diff_1
                        --,close_hist_diff_1
                        ,close_rate as index_close_rate
                    FROM
                        dwd_feat_incr_global_index
                    WHERE
                        date BETWEEN '{args.date_start}' AND '{args.date_end}'
                        AND full_code='000001.sh'
                        """, con=conn.engine)  # 2023-01-01, 2024-01-01
    
        indicators_df = pd.read_sql(f"""
                    SELECT
                        primary_key
                        ,rsi
                        ,cci
                        ,wr
                        ,vwap
                        ,ad 
                        ,mom 
                        ,atr 
                        ,adx 
                        ,plus_di
                        ,minus_di
                        ,mfi
                        ,upper_band
                        ,middle_band
                        ,lower_band
                        ,kdj_fastk
                        ,kdj_fastd
                    FROM
                        dwd_feat_incr_indicators
                    WHERE
                        date BETWEEN '{args.date_start}' AND '{args.date_end}'
                        """, con=conn.engine)
        high_frequency_df = pd.read_sql(f"""
                    SELECT
                        primary_key
                        ,_5min_vwap
                        ,_10pct_5min_low
                        ,_20pct_5min_low
                        ,_80pct_5min_high
                        ,_90pct_5min_high
                    FROM
                        dwd_feat_incr_high_frequency_5minute
                    WHERE
                        date BETWEEN '{args.date_start}' AND '{args.date_end}'
                        """, con=conn.engine)
    # log e 更容易学习特征
# =============================================================================
#     columns_to_transform = ['volume_slope_12_26_9', 'turnover_slope_12_26_9', 'turnover_pct_slope_12_26_9',
#                'volume_hist_12_26_9', 'turnover_hist_12_26_9', 'turnover_pct_hist_12_26_9',
#                'volume_diff_1', 'volume_diff_5', 'volume_diff_30',
#                'turnover_diff_1', 'turnover_diff_5', 'turnover_diff_30',
#                'turnover_hist_diff_1', 'volume_hist_diff_1']
#     stock_daily_df[columns_to_transform] = stock_daily_df[columns_to_transform].map(utils_math.log_e)
# =============================================================================

    df = df.drop_duplicates('primary_key', keep='first') # 不去重会导致数据混入测试集
    
    df = pd.merge(df, index_df, on='date', how='left')
    
    wide_df = df.groupby('full_code').apply(_apply_full_code)
    # df['based_index_class'].value_counts()
    
    # 指定需要loge的列
    columns_to_apply = ['volume','turnover','pe_ttm', 'ps_ttm', 'pcf_ttm', 'pb_mrq']
    wide_df[columns_to_apply] = wide_df[columns_to_apply].apply(utils_math.log_e)
    
    wide_df = pd.merge(wide_df, macd_df, on='primary_key', how='left')
    
    alpha_list = [x for x in alpha_df.columns if 'alpha' in x]
    alpha_df = alpha_df[['primary_key'] + alpha_list]
    wide_df = pd.merge(wide_df, alpha_df, on='primary_key', how='left')

    #wide_df = pd.merge(wide_df, flow_df, on='primary_key', how='left')
    
    wide_df = pd.merge(wide_df, indicators_df, on='primary_key', how='left')
    
    high_frequency_df = high_frequency_df[~((high_frequency_df._5min_vwap==0)|
                                          (high_frequency_df._10pct_5min_low==0)|
                                          (high_frequency_df._20pct_5min_low==0)|
                                          (high_frequency_df._80pct_5min_high==0)|
                                          (high_frequency_df._90pct_5min_high==0))]
    high_frequency_df = high_frequency_df.rename(columns={'_5min_vwap':'y_5min_vwap',
                                                          '_10pct_5min_low':'y_10pct_5min_low',
                                                          '_20pct_5min_low':'y_20pct_5min_low',
                                                          '_80pct_5min_high':'y_80pct_5min_high',
                                                          '_90pct_5min_high':'y_90pct_5min_high',
                                                          })
    wide_df = pd.merge(wide_df, high_frequency_df, on='primary_key', how='left')
    
    utils_data.output_database_large(wide_df,
                                     filename='das_wide_incr_train',
                                     if_exists='replace',
                                     )
    
# =============================================================================
#     utils_data.output_database(wide_df,
#                                filename='das_wide_incr_train',
#                                if_exists='replace')
# =============================================================================

# =============================================================================
# wide_df.isna().sum().to_dict()
# Out[5]: 
# {'primary_key': 0,
#  'adj_type': 0,
#  'freq': 0,
#  'date': 0,
#  'time': 0,
#  'board_type': 0,
#  'full_code': 0,
#  'asset_code': 0,
#  'market_code': 0,
#  'code_name': 0,
#  'price_limit_rate': 0,
#  'open': 0,
#  'high': 0,
#  'low': 0,
#  'close': 0,
#  'prev_close': 0,
#  'volume': 0,
#  'turnover': 0,
#  'turnover_pct': 0,
#  'is_trade': 0,
#  'chg_rel': 0,
#  'pe_ttm': 0,
#  'ps_ttm': 0,
#  'pcf_ttm': 0,
#  'pb_mrq': 0,
#  'is_st': 0,
#  'insert_timestamp': 0,
#  'source_table': 0,
#  'amplitude': 0,
#  'price_chg': 0,
#  'avg_price': 0,
#  'limit_up': 0,
#  'limit_down': 0,
#  'is_limit_up': 0,
#  'is_limit_down': 0,
#  'is_flat_price': 0,
#  'is_limit_up_prev': 0,
#  'is_limit_down_prev': 0,
#  'is_flat_price_prev': 0,
#  'prev_date': 0,
#  'next_date': 0,
#  'date_diff_prev': 0,
#  'date_diff_next': 0,
#  'date_week': 0,
#  'board_primary_key': 0,
#  'index_close': 0,
#  'index_volume': 0,
#  'index_turnover': 0,
#  'index_turnover_pct': 0,
#  'index_close_diff_1': 0,
#  'index_close_diff_5': 0,
#  'index_close_diff_30': 0,
#  'index_volume_diff_1': 0,
#  'index_volume_diff_5': 0,
#  'index_volume_diff_30': 0,
#  'index_turnover_diff_1': 0,
#  'index_turnover_diff_5': 0,
#  'index_turnover_diff_30': 0,
#  'index_close_rate': 0,
#  'high_rate': 0,
#  'low_rate': 0,
#  'open_rate': 0,
#  'close_rate': 0,
#  'avg_price_rate': 0,
#  'next_high_rate': 0,
#  'next_low_rate': 0,
#  'next_open_rate': 0,
#  'next_close_rate': 0,
#  'next_index_close_rate': 0,
#  'next_relative_close_rel': 0,
#  'next_based_index_class': 0,
#  'close_slope_12_26_9': 0,
#  'volume_slope_12_26_9': 0,
#  'turnover_slope_12_26_9': 0,
#  'turnover_pct_slope_12_26_9': 0,
#  'close_acceleration_12_26_9': 0,
#  'volume_acceleration_12_26_9': 0,
#  'turnover_acceleration_12_26_9': 0,
#  'turnover_pct_acceleration_12_26_9': 0,
#  'close_hist_12_26_9': 0,
#  'volume_hist_12_26_9': 0,
#  'turnover_hist_12_26_9': 0,
#  'turnover_pct_hist_12_26_9': 0,
#  'close_diff_1': 0,
#  'close_diff_5': 0,
#  'close_diff_30': 0,
#  'volume_diff_1': 0,
#  'volume_diff_5': 0,
#  'volume_diff_30': 0,
#  'turnover_diff_1': 0,
#  'turnover_diff_5': 0,
#  'turnover_diff_30': 0,
#  'turnover_hist_diff_1': 0,
#  'volume_hist_diff_1': 0,
#  'close_hist_diff_1': 0,
#  'alpha001': 0,
#  'alpha002': 0,
#  'alpha003': 0,
#  'alpha004': 0,
#  'alpha005': 0,
#  'alpha006': 0,
#  'alpha007': 0,
#  'alpha008': 0,
#  'alpha009': 0,
#  'alpha010': 0,
#  'alpha011': 0,
#  'alpha012': 0,
#  'alpha013': 0,
#  'alpha014': 0,
#  'alpha015': 0,
#  'alpha016': 0,
#  'alpha017': 0,
#  'alpha018': 0,
#  'alpha019': 0,
#  'alpha020': 0,
#  'alpha021': 0,
#  'alpha022': 0,
#  'alpha023': 0,
#  'alpha024': 0,
#  'alpha025': 0,
#  'alpha026': 0,
#  'alpha027': 0,
#  'alpha028': 0,
#  'alpha029': 0,
#  'alpha030': 0,
#  'alpha031': 0,
#  'alpha032': 0,
#  'alpha033': 0,
#  'alpha034': 0,
#  'alpha035': 0,
#  'alpha036': 0,
#  'alpha037': 0,
#  'alpha038': 0,
#  'alpha039': 0,
#  'alpha040': 0,
#  'alpha041': 0,
#  'alpha042': 0,
#  'alpha043': 0,
#  'alpha044': 0,
#  'alpha045': 0,
#  'alpha046': 0,
#  'alpha047': 0,
#  'alpha049': 0,
#  'alpha050': 0,
#  'alpha051': 0,
#  'alpha052': 0,
#  'alpha053': 0,
#  'alpha054': 0,
#  'alpha055': 0,
#  'alpha057': 0,
#  'alpha060': 0,
#  'alpha061': 965886,
#  'alpha062': 0,
#  'alpha064': 0,
#  'alpha065': 0,
#  'alpha066': 0,
#  'alpha068': 0,
#  'alpha071': 0,
#  'alpha072': 0,
#  'alpha073': 0,
#  'alpha074': 0,
#  'alpha075': 965886,
#  'alpha077': 0,
#  'alpha078': 0,
#  'alpha081': 0,
#  'alpha083': 0,
#  'alpha084': 0,
#  'alpha085': 0,
#  'alpha086': 0,
#  'alpha088': 0,
#  'alpha092': 0,
#  'alpha094': 0,
#  'alpha095': 965886,
#  'alpha096': 0,
#  'alpha098': 0,
#  'alpha099': 0,
#  'alpha101': 0,
#  'rsi': 0,
#  'cci': 0,
#  'wr': 0,
#  'vwap': 0,
#  'ad': 0,
#  'mom': 0,
#  'atr': 0,
#  'adx': 0,
#  'plus_di': 0,
#  'minus_di': 0,
#  'mfi': 0,
#  'upper_band': 0,
#  'middle_band': 0,
#  'lower_band': 0,
#  'kdj_fastk': 0,
#  'kdj_fastd': 0}
# =============================================================================
