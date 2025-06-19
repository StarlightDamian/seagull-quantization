# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 23:44:18 2024

@author: awei
lightgbm_train_class


"""
import os
import argparse

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, fbeta_score, confusion_matrix
#import lightgbm as lgb
#import joblib


from seagull.settings import PATH
from seagull.utils import utils_database, utils_log
import lightgbm_base
import rolling_cv

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')

#board_primary_key
#df.drop_duplicates('board_primary_key',keep='first')[['board_type','is_limit_up_prev','is_limit_down_prev']]
#427892
#df.groupby('board_primary_key').apply(lambda x:x.shape[0])

# =============================================================================
# def board(df):
#     df['num'] = df.shape[0]
#     df = df.drop_duplicates('board_primary_key',keep='first')
#     return df[['board_type','price_limit_rate','is_limit_up_prev','is_limit_down_prev','num']]
# 
# result_df = df.groupby('board_primary_key').apply(board).reset_index(drop=True)
# result_df.reset_index(drop=True).sort_values(by=['board_type','price_limit_rate'],ascending=False)
# #,'is_limit_up_prev','is_limit_down_prev'
# =============================================================================

if __name__ == '__main__':
# =============================================================================
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--date_start', type=str, default='2024-01-01', help='Start time for backtesting')
#     parser.add_argument('--date_end', type=str, default='2024-06-01', help='End time for backtesting')
#     args = parser.parse_args()
#     
#     with utils_database.engine_conn("POSTGRES") as conn:
#         raw_df = pd.read_sql(f"SELECT * FROM das_wide_incr_train WHERE date >= '{args.date_start}' AND date < '{args.date_end}'", con=conn.engine)
#         # raw_df = pd.read_sql("select * from das_wide_incr_train where board_type='主板'", con=conn.engine)
#         # raw_df = pd.read_sql("select *  from das_wide_incr_train where board_type not in ('北交所','ETF') ", con=conn.engine)
#         
# =============================================================================
    raw_df = pd.read_feather(f'{PATH}/_file/das_wide_incr_train.feather')
    
    # 清洗脏数据
    raw_df = raw_df[(raw_df.high <= raw_df.limit_up)&(raw_df.low >= raw_df.limit_down)]
    raw_df = raw_df[~((raw_df.next_high_rate.apply(np.isinf))|(raw_df.next_low_rate.apply(np.isinf)))]

    ohlc_features = ['open_rate', 'high_rate', 'low_rate', 'close_rate', 'volume', 'turnover', 'turnover_pct',
                     'price_limit_rate','board_type', 'date_diff_prev', 'date_diff_next', 'date_week','is_flat_price_prev','is_limit_down_prev',
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
           'small_inflow_hist_diff_1',
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
    
    features = ohlc_features +\
               fundamental_features +\
               label_features +\
               macd_features +\
               alpha_features +\
               index_features +\
               indicators_features
               #flow_features +\
    # 特征和目标
    x = raw_df[features+['date']]#raw_df[features]#
    y_class = raw_df[['next_based_index_class','date']]#raw_df['next_based_index_class']#
    
    # 处理异常值
    y_class = y_class.replace([np.inf, -np.inf, np.nan], 0)
    x = x.replace([np.inf, -np.inf, np.nan], 1)  #.fillna(1)  # debug:这个错误是由于输入数据中包含无穷值 (inf)、负无穷值 (-inf)、或超出 float64 类型范围的值导致的
    constant_features = x.columns[x.nunique() == 1]  # 筛选出所有值相同的列
    logger.warning(f'相同值特征: {constant_features}')  # ['alpha019', 'alpha027', 'alpha039']
    x = x.loc[:, x.nunique() > 1]
    
    # 特征选择
    categorical_features = ['board_type','date_week']
    numeric_features = list(set(features)-set(constant_features)-set(categorical_features))
    # x[numeric_features] = x[numeric_features].astype(float)
    
    # 划分训练和测试集
    #x_train, x_test, y_class_train, y_class_test = train_test_split(
     #   x, y_class, test_size=0.2, random_state=42)
    
    date_series = sorted(raw_df['date'].unique()) 
    train_days = int(len(date_series)*0.75) # 60
    split_df = rolling_cv.rolling_window_split(date_series, train_days=train_days, gap_days=2, val_rate=0.2, n_splits=2)
    
    x_train = x[(x.date >= split_df.loc[0,'train_start'])
                &(x.date < split_df.loc[0,'train_end'])]
    x_test = x[(x.date >= split_df.loc[0,'val_start'])
                &(x.date < split_df.loc[0,'val_end'])]
    y_class_train = y_class[(y_class.date >= split_df.loc[0,'train_start'])
                &(y_class.date < split_df.loc[0,'train_end'])]
    y_class_test = y_class[(y_class.date >= split_df.loc[0,'val_start'])
                &(y_class.date < split_df.loc[0,'val_end'])]
    del x_train['date']
    del x_test['date']
    del y_class_train['date']
    del y_class_test['date']
    
    # 初始化类
    pipeline = lightgbm_base.StockModelPipeline()
    
    # 分类
    param_grid_class = {
    'classifier__objective': ['multiclass'],  # 三分类任务
    'classifier__num_class': [3],  # 类别数
    'classifier__boosting_type': ['gbdt'],  # 梯度提升决策树
    'classifier__metric': ['multi_logloss'],  # 多分类交叉熵
    'classifier__num_leaves': [255],  # 树的叶子数
    'classifier__max_depth': [8],  # 最大深度
    'classifier__learning_rate': [0.12],  # 学习率
    'classifier__n_estimators': [1200], # [1200],  # 树的数量
    # 'classifier__min_child_samples': [10, 20, 30],  # 每个叶子节点的最小样本数
    # 'classifier__subsample': [0.7, 0.8, 1.0],  # 样本采样比例
    # 'classifier__colsample_bytree': [0.7, 0.8, 1.0],  # 每棵树的特征采样比例
    }
    
    pipeline.fit_class_model(x_train,
                             y_class_train,
                             numeric_features,
                             categorical_features,
                             param_grid_class)
    y_class_pred = pipeline.predict_class(x_test)
    
    # 打印最佳参数和对应的 F05 score
    logger.info("Best F0.5 score: ", pipeline.grid_search.best_score_)
    logger.info("Best parameters found: ", pipeline.grid_search.best_params_)
    
    # 评估
    print("Classification Report:")
    conf_matrix = confusion_matrix(y_class_test, y_class_pred, labels=[-2, -1, 0, 1, 2])
    print(conf_matrix)
    #print(classification_report(y_class_test, y_class_pred))
    class_report_df = pd.DataFrame(classification_report(y_class_test, y_class_pred, output_dict=True)).T
    class_report_df=class_report_df.head(-3) # del 'accuracy', 'macro avg', 'weighted avg'
    fbeta = fbeta_score(y_class_test, y_class_pred, beta=0.5, average=None)
    class_report_df['f05-score'] = fbeta
    print(class_report_df)
    
    result_class = pd.DataFrame([y_class_test.values, y_class_pred]).T
    result_class.columns=['class_real','class_pred']
    result_class[result_class.class_real==1]
    result_class[result_class.class_pred==1].head(40)
    print(result_class.head(40))
    result_class[result_class.class_real==result_class.class_pred]
    
    # 保存模型
    #pipeline.save_models('cluster.pkl', 'class.pkl', 'reg.pkl')
    
    # 加载模型
    #pipeline.load_models('cluster.pkl', 'class.pkl', 'reg.pkl')

# =============================================================================
# Classification Report:
# [[    0     0     0     0     0]
#  [    0  8421 24900  1711     0]
#  [    0  4848 89311  1613     0]
#  [    0  5510 27105  3600     0]
#  [    0     0     0     0     0]]
#     precision    recall  f1-score  support  f05-score
# -1   0.448426  0.240380  0.312984  35032.0   0.382258
# 0    0.631995  0.932538  0.753400  95772.0   0.675538
# 1    0.519931  0.099406  0.166902  36215.0   0.281642
# Regression RMSE:
# 3.0655
# =============================================================================
# =============================================================================log_e量价
# [[    0     0     0     0     0]
#  [    0  8333 24255  1902     0]
#  [    0  4464 89475  1998     0]
#  [    0  5230 26732  4625     0]
#  [    0     0     0     0     0]]
#     precision    recall  f1-score  support  f05-score
# -1   0.462251  0.241606  0.317345  34490.0   0.390861
# 0    0.637005  0.932643  0.756983  95937.0   0.680123
# 1    0.542522  0.126411  0.205045  36587.0   0.327146
# Regression RMSE:
# 9.5788
# =============================================================================
# =============================================================================
# [[    0     0     0     0     0]
#  [    0  8792 24938  1852     0]
#  [    0  4912 88755  1990     0]
#  [    0  5398 25946  4414     0]
#  [    0     0     0     0     0]]
#     precision    recall  f1-score  support  f05-score
# -1   0.460266  0.247091  0.321557  35582.0   0.392535
# 0    0.635603  0.927846  0.754411  95657.0   0.678334
# 1    0.534641  0.123441  0.200573  35758.0   0.320869
# =============================================================================
# =============================================================================
# [[    0     0     0     0     0]
#  [    0  8893 24883  1766     0]
#  [    0  4705 90373  1887     0]
#  [    0  5578 26989  4430     0]
#  [    0     0     0     0     0]]
#     precision    recall  f1-score  support  f05-score
# -1   0.463757  0.250211  0.325048  35542.0   0.396139
# 0    0.635333  0.932017  0.755596  96965.0   0.678532
# 1    0.548064  0.119739  0.196539  36997.0   0.319491
# =============================================================================
# =============================================================================
# [[    0     0     0     0     0]
#  [    0     0     0     0     0]
#  [    0     0 12770  9048  6066]
#  [    0     0  6248 16365  5474]
#  [    0     0  9578 10148  8900]]
#      precision    recall  f1-score  support  f05-score
# 0.0   0.446566  0.457969  0.452195  27884.0   0.448801
# 1.0   0.460195  0.582654  0.514235  28087.0   0.480388
# 2.0   0.435421  0.310906  0.362777  28626.0   0.403131
# =============================================================================

# =============================================================================
# [[     0      0      0      0      0]
#  [     0  14174  34808   1706      0]
#  [     0   8665 110587   1694      0]
#  [     0   9475  34307   3139      0]
#  [     0      0      0      0      0]]
#     precision    recall  f1-score   support  f05-score
# -1   0.438633  0.279632  0.341534   50688.0   0.393845
# 0    0.615391  0.914350  0.735658  120946.0   0.658449
# 1    0.480043  0.066900  0.117434   46921.0   0.214773
# =============================================================================0
# =============================================================================
# [[     0      0      0      0      0]
#  [     0  14325  34909   1772      0]
#  [     0   8638 110205   1640      0]
#  [     0   9490  34292   3334      0]
#  [     0      0      0      0      0]]
#     precision    recall  f1-score   support  f05-score
# -1   0.441408  0.280849  0.343282   51006.0   0.396117
# 0    0.614277  0.914693  0.734972  120483.0   0.657464
# 1    0.494219  0.070762  0.123798   47116.0   0.224966
# =============================================================================

# =============================================================================
# [[     0      0      0      0      0]
#  [     0  13825  34165   1750      0]
#  [     0   8511 110677   1798      0]
#  [     0   9316  35005   3499      0]
#  [     0      0      0      0      0]]
#     precision    recall  f1-score   support  f05-score
# -1   0.436781  0.277945  0.339714   49740.0   0.391981
# 0    0.615395  0.914792  0.735804  120986.0   0.658498
# 1    0.496523  0.073170  0.127545   47820.0   0.230173
# =============================================================================
# =============================================================================
# Classification Report:
# [[    0     0     0     0     0]
#  [    0  4505 13045   913     0]
#  [    0  2646 44398   999     0]
#  [    0  3094 12983  1899     0]
#  [    0     0     0     0     0]]
#     precision    recall  f1-score  support  f05-score
# -1   0.439727  0.244002  0.313850  18463.0   0.378934
# 0    0.630421  0.924130  0.749529  48043.0   0.673213
# 1    0.498294  0.105641  0.174324  17976.0   0.285822
# =============================================================================
# =============================================================================
# [[    0     0     0     0     0]
#  [    0  4473 12989   960     0]
#  [    0  2612 44608  1056     0]
#  [    0  2890 13052  1971     0]
#  [    0     0     0     0     0]]
#     precision    recall  f1-score  support  f05-score
# -1   0.448421  0.242808  0.315033  18422.0   0.383475
# 0    0.631403  0.924020  0.750187  48276.0   0.674098
# 1    0.494357  0.110032  0.180000  17913.0   0.291043
# =============================================================================
# =============================================================================alpha
# Classification Report:
# [[    0     0     0     0     0]
#  [    0  5722 11182  1331     0]
#  [    0  2710 43981  1735     0]
#  [    0  2468 11544  3801     0]
#  [    0     0     0     0     0]]
#     precision    recall  f1-score  support  f05-score
# -1   0.524954  0.313792  0.392792  18235.0   0.462683
# 0    0.659316  0.908210  0.764003  48426.0   0.697549
# 1    0.553517  0.213383  0.308023  17813.0   0.419712
# =============================================================================
# =============================================================================pre
# Classification Report:
# [[    0     0     0     0     0]
#  [    0  5580 11158  1375     0]
#  [    0  2715 43170  1696     0]
#  [    0  2383 11446  3820     0]
#  [    0     0     0     0     0]]
#     precision    recall  f1-score  support  f05-score
# -1   0.522570  0.308066  0.387621  18113.0   0.458693
# 0    0.656338  0.907295  0.761678  47581.0   0.694773
# 1    0.554346  0.216443  0.311328  17649.0   0.422445
# =============================================================================
# =============================================================================
# Classification Report:
# [[    0     0     0     0     0]
#  [    0  5642 11365  1316     0]
#  [    0  2691 43952  1683     0]
#  [    0  2531 11550  3856     0]
#  [    0     0     0     0     0]]
#     precision    recall  f1-score  support  f05-score
# -1   0.519330  0.307919  0.386610  18323.0   0.456628
# 0    0.657305  0.909490  0.763102  48326.0   0.695897
# 1    0.562509  0.214975  0.311068  17937.0   0.425072
# =============================================================================