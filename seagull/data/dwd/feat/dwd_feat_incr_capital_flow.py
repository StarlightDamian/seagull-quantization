# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 20:57:00 2024

@author: awei
(dwd_feat_incr_capital_flow_temp)
临时表，用于测试资金流动特征的效果，稳定后会在ods->dwd层直接清洗对应数据
需要区分adj_type和freq

字段	类型	注释	说明
stock_code	string	代码	600001
trade_date	date	交易日期	2023-09-13
main_net_inflow	decimal	主力资金净流入(元)	2526394.0
max_net_inflow	decimal	特大单净流入(元)	2526394.0
lg_net_inflow	decimal	大单净流入(元)	2526394.0
mid_net_inflow	decimal	中单净流入(元)	2526394.0
sm_net_inflow	decimal	小单净流入(元)	2526394.0
"""
import os

import pandas as pd

from seagull.settings import PATH
from seagull.utils import utils_database, utils_log, utils_data, utils_math
from seagull.data import ods_info_incr_adata_stock_base_api
from data.dwd.feat import dwd_feat_incr_macd

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')

def pipeline(date_start, date_end):
    ...

if __name__ == '__main__':
    with utils_database.engine_conn("POSTGRES") as conn:
        capital_flow_df = pd.read_sql("ods_feat_incr_adata_capital_flow", con=conn.engine)
    
    capital_flow_df = capital_flow_df.rename(columns={'stock_code': 'asset_code',
                                                      'trade_date': 'date',
                                                      'main_net_inflow': 'main_inflow',
                                                      'max_net_inflow': 'ultra_large_inflow',
                                                      'lg_net_inflow': 'large_inflow',
                                                      'mid_net_inflow': 'medium_inflow',
                                                      'sm_net_inflow': 'small_inflow',
                                                      })

    #capital_flow_df = capital_flow_df.fillna(0)  # 很多null
    capital_flow_df['freq'] = 'd'
    capital_flow_df['adj_type'] = 'pre'   # adjustment_type as adj_type in ['None', 'pre', 'post']
    capital_flow_df = ods_info_incr_adata_stock_base_api.associated_primary_key(capital_flow_df)  # 必备字段 df.columns = ['date', 'freq', 'adj_type']
    capital_flow_df = capital_flow_df.drop_duplicates('primary_key', keep='first')
    
    #capital_flow_values = capital_flow_df[['main_inflow', 'ultra_large_inflow', 'large_inflow', 'medium_inflow', 'small_inflow']].astype(float).values
    #capital_flow_log10_arr = signed_log10(capital_flow_values)
    columns_to_transform = ['main_inflow', 'ultra_large_inflow', 'large_inflow', 'medium_inflow', 'small_inflow']
    capital_flow_df[columns_to_transform] = capital_flow_df[columns_to_transform].astype(float)  # 之前是object
    capital_flow_df['main_small_net_inflow'] = capital_flow_df['main_inflow'] - capital_flow_df['small_inflow']
    columns_to_transform += ['main_small_net_inflow']
    
    capital_flow_df = dwd_feat_incr_macd.pipeline(capital_flow_df,
                                                  columns=columns_to_transform,
                                                  numeric_columns=columns_to_transform,
                                                  )
    
    
    capital_flow_df[columns_to_transform] = capital_flow_df[columns_to_transform].apply(utils_math.log_e)
    #dwd_capital_flow = pd.DataFrame(capital_flow_df, columns=['log10_main_inflow', 'log10_ultra_large_inflow', 'log10_large_inflow', 'log10_medium_inflow', 'log10_small_inflow'])
    capital_flow_df = capital_flow_df.rename(columns={'main_inflow':'loge_main_inflow',
                                                       'ultra_large_inflow':'loge_ultra_large_inflow',
                                                       'large_inflow':'loge_large_inflow',
                                                       'medium_inflow':'loge_medium_inflow',
                                                       'small_inflow':'loge_small_inflow',
                                                       'main_small_net_inflow':'loge_main_small_net_inflow'})
    

    
    columns_to_transform = ['loge_main_inflow', 'loge_ultra_large_inflow', 'loge_large_inflow', 'loge_medium_inflow', 'loge_small_inflow']
    capital_flow_df[columns_to_transform] = capital_flow_df[columns_to_transform].round(5)
    
    #capital_flow_df = capital_flow_df[['primary_key', 'full_code', 'asset_code', 'market_code', 'date', 'time', 'freq']]
    #dwd_capital_flow = pd.concat([capital_flow_df, dwd_capital_flow], axis=1)
    conn = utils_database.psycopg2_conn()
    utils_data.output_database_large(capital_flow_df,
                               filename='dwd_feat_incr_capital_flow',
                               conn=conn,
                               if_exists='replace',
                               )

