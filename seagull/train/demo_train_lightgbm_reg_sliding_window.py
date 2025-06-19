# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 16:30:04 2025

@author: awei

demo_train_lightgbm_reg_sliding_window
"""

import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from sklearn.metrics import classification_report, fbeta_score, confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
#import warnings


from seagull.settings import PATH
from seagull.utils import utils_database, utils_log, utils_thread,utils_math
from feature import vwap, max_drawdown
log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

if __name__ == '__main__':
    # 生成示例时间序列数据
    #rew_df = pd.read_feather(f'{PATH}/data/das_wide_incr_train.feather')
    #raw_df = rew_df.groupby('full_code').apply(vwap.daily_vwap, window=10)
    #raw_df = raw_df.rename(columns={'y_vwap_rate': 'y_10d_vwap_rate'})
    #raw_df.reset_index(drop=True).to_feather(f'{PATH}/data/das_wide_incr_train_10d_vwap.feather')
    raw_df = pd.read_feather(f'{PATH}/data/das_wide_incr_train_10d_vwap.feather')
    #raw_df = raw_df.groupby('full_code').apply(max_drawdown.calculate_max_drawdown, column_name='10d_max_dd', window=10)
    #raw_df.reset_index(drop=True).to_feather(f'{PATH}/data/das_wide_incr_train_10d_vwap_max_dd.feather')
    #raw_df = pd.read_feather(f'{PATH}/data/das_wide_incr_train_10d_vwap_max_dd.feather')

    # 数据处理
   # alpha_features = ['alpha001', 'alpha002', 'alpha003', 'alpha004', 'alpha005', 'alpha006', 'alpha007', 'alpha008', 'alpha009', 'alpha010', 'alpha011', 'alpha012', 'alpha013', 'alpha014', 'alpha015', 'alpha016', 'alpha017', 'alpha018', 'alpha019', 'alpha020', 'alpha021', 'alpha022', 'alpha023', 'alpha024', 'alpha025', 'alpha026', 'alpha027', 'alpha028', 'alpha029', 'alpha030', 'alpha031', 'alpha032', 'alpha033', 'alpha034', 'alpha035', 'alpha036', 'alpha037', 'alpha038', 'alpha039', 'alpha040', 'alpha041', 'alpha042', 'alpha043', 'alpha044', 'alpha045', 'alpha046', 'alpha047', 'alpha049', 'alpha050', 'alpha051', 'alpha052', 'alpha053', 'alpha054', 'alpha055', 'alpha057', 'alpha060', 'alpha061', 'alpha062', 'alpha064', 'alpha065', 'alpha066', 'alpha068', 'alpha071', 'alpha072', 'alpha073', 'alpha074', 'alpha075', 'alpha077', 'alpha078', 'alpha081', 'alpha083', 'alpha084', 'alpha085', 'alpha086', 'alpha088', 'alpha092', 'alpha094', 'alpha095', 'alpha096', 'alpha098', 'alpha099', 'alpha101']


    
    # 清洗脏数据
    raw_df = raw_df[(raw_df.high <= raw_df.limit_up)&(raw_df.low >= raw_df.limit_down)]
    raw_df = raw_df[~((raw_df.next_high_rate.apply(np.isinf))|(raw_df.next_low_rate.apply(np.isinf)))]
    
    ohlc_features = ['open_rate', 'high_rate', 'low_rate', 'close_rate', 'volume', 'turnover', 'turnover_pct',
                     'price_limit_rate','board_type', 'date_diff_prev', 'date_diff_next', 'date_week','is_limit_down_prev',
                     'is_limit_up_prev']
    fundamental_features = ['chg_rel', 'pe_ttm', 'ps_ttm', 'pcf_ttm', 'pb_mrq']
    # =============================================================================
    #     # 多标签进行onehot， 先置空加快计算
    #     train_df = onehot(train_df, label_df[['full_code', 'label']])
    #     label_features =  [x for x in train_df.columns if ('label_' in x) and ('label_昨日' not in x)]#[]#
    # =============================================================================
    label_features = []
    macd_features = ['close_slope_12_26_9'
                ,'volume_slope_12_26_9'
                ,'turnover_slope_12_26_9'
                ,'turnover_pct_slope_12_26_9'
                ,'close_acceleration_12_26_9'
                ,'volume_acceleration_12_26_9'
                ,'turnover_acceleration_12_26_9'
                ,'turnover_pct_acceleration_12_26_9'
                ,'close_hist_12_26_9'
                ,'volume_hist_12_26_9'
                ,'turnover_hist_12_26_9'
                ,'turnover_pct_hist_12_26_9'
                ,'close_diff_1'
                ,'close_diff_5'
                ,'close_diff_30'
                ,'volume_diff_1'
                ,'volume_diff_5'
                ,'volume_diff_30'
                ,'turnover_diff_1'
                ,'turnover_diff_5'
                ,'turnover_diff_30'
                ,'turnover_hist_diff_1'
                ,'volume_hist_diff_1'
                ,'close_hist_diff_1']
    flow_features = ['loge_main_inflow', 'loge_ultra_large_inflow',
           'loge_large_inflow', 'loge_medium_inflow', 'loge_small_inflow','loge_main_small_net_inflow',
           'main_inflow_slope_12_26_9', 'ultra_large_inflow_slope_12_26_9',
           'large_inflow_slope_12_26_9', 'medium_inflow_slope_12_26_9',
           'small_inflow_slope_12_26_9', 'main_inflow_acceleration_12_26_9',
           'ultra_large_inflow_acceleration_12_26_9',
           'large_inflow_acceleration_12_26_9',
           'medium_inflow_acceleration_12_26_9',
           'small_inflow_acceleration_12_26_9', 'main_inflow_hist_12_26_9',
           'ultra_large_inflow_hist_12_26_9', 'large_inflow_hist_12_26_9',
           'medium_inflow_hist_12_26_9', 'small_inflow_hist_12_26_9',
           'main_inflow_diff_1', 'main_inflow_diff_5', 'main_inflow_diff_30',
           'ultra_large_inflow_diff_1', 'ultra_large_inflow_diff_5',
           'ultra_large_inflow_diff_30', 'large_inflow_diff_1',
           'large_inflow_diff_5', 'large_inflow_diff_30', 'medium_inflow_diff_1',
           'medium_inflow_diff_5', 'medium_inflow_diff_30', 'small_inflow_diff_1',
           'small_inflow_diff_5', 'small_inflow_diff_30',
           'main_inflow_hist_diff_1', 'ultra_large_inflow_hist_diff_1',
           'large_inflow_hist_diff_1', 'medium_inflow_hist_diff_1',
           'small_inflow_hist_d-0iff_1',
           'main_small_net_inflow_slope_12_26_9',
           'main_small_net_inflow_acceleration_12_26_9',
           'main_small_net_inflow_hist_12_26_9',
           'main_small_net_inflow_diff_1',
           'main_small_net_inflow_diff_5',
           'main_small_net_inflow_diff_30',
           'main_small_net_inflow_hist_diff_1']
    alpha_features = ['alpha001', 'alpha002', 'alpha003', 'alpha004', 'alpha005', 'alpha006', 'alpha007', 'alpha008', 'alpha009', 'alpha010', 'alpha011', 'alpha012', 'alpha013', 'alpha014', 'alpha015', 'alpha016', 'alpha017', 'alpha018', 'alpha019', 'alpha020', 'alpha021', 'alpha022', 'alpha023', 'alpha024', 'alpha025', 'alpha026', 'alpha027', 'alpha028', 'alpha029', 'alpha030', 'alpha031', 'alpha032', 'alpha033', 'alpha034', 'alpha035', 'alpha036', 'alpha037', 'alpha038', 'alpha039', 'alpha040', 'alpha041', 'alpha042', 'alpha043', 'alpha044', 'alpha045', 'alpha046', 'alpha047', 'alpha049', 'alpha050', 'alpha051', 'alpha052', 'alpha053', 'alpha054', 'alpha055', 'alpha057', 'alpha060', 'alpha061', 'alpha062', 'alpha064', 'alpha065', 'alpha066', 'alpha068', 'alpha071', 'alpha072', 'alpha073', 'alpha074', 'alpha075', 'alpha077', 'alpha078', 'alpha081', 'alpha083', 'alpha084', 'alpha085', 'alpha086', 'alpha088', 'alpha092', 'alpha094', 'alpha095', 'alpha096', 'alpha098', 'alpha099', 'alpha101']
    index_features = ['index_close_diff_1', 'index_close_diff_5', 'index_close_diff_30', 'index_volume_diff_1','index_volume_diff_5',
                      'index_volume_diff_30', 'index_turnover_diff_1', 'index_turnover_diff_5', 'index_turnover_diff_30','index_close_rate',
                      'index_close','index_volume','index_turnover','index_turnover_pct']
    indicators_features = ['rsi','cci', 'wr', 'vwap', 'ad', 'mom', 'atr', 'adx', 'plus_di', 'minus_di', 'mfi', 'upper_band', 'middle_band', 'lower_band', 'kdj_fastk','kdj_fastd']
    
    categorical_features = ['board_type','date_week','alpha061','alpha062','alpha064','alpha065','alpha068',
                            'alpha074','alpha075','alpha081','alpha086','alpha095','alpha099']
    #alpha068
    features = index_features +\
               indicators_features +\
               macd_features +\
               alpha_features +\
               ohlc_features +\
               fundamental_features +\
               label_features
               #flow_features +\
    numeric_features = list(set(features)-
                            set(['alpha061','alpha062','alpha064','alpha065','alpha068','alpha074','alpha075',
                                 'alpha081','alpha086','alpha095','alpha099']))
    
    #grouped = raw_df.groupby('full_code')
    #raw_df = utils_thread.thread(grouped, calculate_close_rate, max_workers=8)
    raw_df[categorical_features] = raw_df[categorical_features].astype('category')
    raw_df = raw_df[~raw_df['y_10d_vwap_rate'].isnull()]
    raw_df.sort_values(by='date', ascending=True, inplace=True, ignore_index=True)
    #y_10d_vwap, next_based_index_class,10d_max_dd
   # raw_df[['y_10d_vwap_rate']] = raw_df[['y_10d_vwap']].div(raw_df['close'], axis=0)
   # raw_df = raw_df[~raw_df.y_10d_vwap_rate.isna()]

    # 特征和目标，y_10d_vwap
    X = raw_df.loc[raw_df.date<='2024-12-02', features]  # raw_df[features+['date']]#
    y = raw_df.loc[raw_df.date<='2024-12-02', 'y_10d_vwap_rate']  # raw_df[['next_based_index_class','date']]#
    X_val = raw_df.loc[raw_df.date>'2024-12-03', features]
    y_val = raw_df.loc[raw_df.date>'2024-12-03', 'y_10d_vwap_rate']
    
    # 存储所有的预测结果
    predictions = []
    params={'objective': 'regression',
            'boosting_type': 'gbdt',#dart, gbdt
            'metric': 'rmse',#multi_error,multi_logloss
            'num_leaves': 127,
            'max_depth':7,
            'learning_rate': 0.05,
            'n_estimators': 1200,#int(200/n_splits),
            'min_gain_to_split':0.1,
            }
    model = lgb.LGBMRegressor(#device='gpu',
                              # gpu_platform_id=0,
                              # gpu_device_id=0,
                               #max_bin=255,
                               verbose=-1,  # 输出
                               #class_weight='balanced',
                               #class_weight={0: 2,
                               #              1: 5,
                               #              2: 2,},
                               early_stopping_rounds=50, 
                               #importance_type='gain',
                               **params
                               )
    window_size = 800_000  # 训练窗口大小
    test_size = 10_000  # 测试窗口大小
    step_size = 40_000  # 滑动步长
    # 滑动窗口训练过程
    for start_idx in range(0, len(X) - window_size - test_size, step_size):
        # 定义训练集和测试集的索引
        train_end_idx = start_idx + window_size
        test_end_idx = train_end_idx + test_size
        
        # 划分训练集和测试集
        X_train, y_train = X[start_idx:train_end_idx], y[start_idx:train_end_idx]
        X_test, y_test = X[train_end_idx:test_end_idx], y[train_end_idx:test_end_idx]
        
        # 训练LightGBM模型
        model.fit(X_train,
                  y_train, 
                  eval_set=[(X_test, y_test)], 
                  eval_metric='rmse',
                  )
        
        # 获取预测值
        y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
        predictions.append(y_pred)
        
        # 输出当前的RMSE评估
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"RMSE for window {start_idx} to {test_end_idx}: {rmse:6f}")
        
    logger.info(f'模型迭代次数: {model.n_iter_}')
    y_val_pred = model.predict(X_val)
    logger.info(f'val rmse {rmse(y_val, y_val_pred):.6f}')
    result_reg = pd.DataFrame([y_val.values, y_val_pred]).T
    result_reg.columns=['y_real','y_pred']
    result_reg = result_reg.round(4)
    result_reg = result_reg.sort_values(by='y_pred' ,ascending=False)
    result_reg.to_csv(f'{PATH}/data/test_reg_10d_max_dd.csv', index=False)
    
    #rmse(y_val.values, y_val_pred)#0.053320363282936774