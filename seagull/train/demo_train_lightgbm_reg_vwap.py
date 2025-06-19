# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 21:04:12 2024

@author: awei
demo_train_lightgbm_reg_vwap
y_10d_vwap_rate
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from sklearn.metrics import classification_report, fbeta_score, confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
#import warnings


from __init__ import path
from utils import utils_database, utils_log, utils_thread,utils_math
from feature import vwap
log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{path}/log/{log_filename}.log')

#warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')

 
def f05_score_lgb(y_true, y_pred):
    """
    Compute F0.5 score for LightGBM
    """
    # LightGBM's `y_pred` contains probabilities, so we need to convert them into binary predictions
    y_pred_binary = (y_pred >= 0.5).astype(int)
    
    # Calculate F0.5 score
    score = fbeta_score(y_true, y_pred_binary, beta=0.5, average='weighted')
    
    # LightGBM expects the result as a tuple (name, score)
    return 'f05_score', score

def f05_score(y_true, y_pred):
    """
    计算 F0.5 score
    F0.5 = (1 + 0.5^2) * (precision * recall) / (0.5^2 * precision + recall)
    """
    return fbeta_score(y_true, y_pred, beta=0.5, average='weighted')
f05_scorer = make_scorer(f05_score)

def calculate_close_rate(df: pd.DataFrame) -> pd.DataFrame:
    # df = df.sort_values(by='date', ascending=True)
    #df[['close_rate']] = df[['close_rate']].shift(-1)
    
    df[numeric_features] = df[numeric_features].apply(utils_math.log_e)
    #df[numeric_features] = winsorize(df[numeric_features])
    #df[numeric_features] = standardize(df[numeric_features])
    return df

def standardize(data):
    """
    标准化处理
    """
    return (data - data.mean()) / data.std()

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

if __name__ == '__main__':
    # 生成示例时间序列数据
    raw_df = pd.read_feather(f'{path}/data/das_wide_incr_train.feather')
    raw_df = raw_df.groupby('full_code').apply(vwap.vwap_pipeline, column_name='y_10d_vwap',window=5)
    #raw_df.reset_index(drop=True).to_feather(f'{path}/data/das_wide_incr_train_10d_vwap.feather')         
    #raw_df = pd.read_feather(f'{path}/data/das_wide_incr_train_10d_vwap.feather')
# =============================================================================
#     alpha_features=['alpha098'
# ,'alpha096'
# ,'alpha094'
# ,'alpha092'
# ,'alpha085'
# ,'alpha084'
# ,'alpha072'
# ,'alpha071'
# ,'alpha066'
# ,'alpha060'
# ,'alpha057'
# ,'alpha054'
# ,'alpha049'
# ,'alpha047'
# ,'alpha046'
# ,'alpha043'
# ,'alpha042'
# ,'alpha041'
# ,'alpha039'
# ,'alpha038'
# ,'alpha037'
# ,'alpha036'
# ,'alpha035'
# ,'alpha033'
# ,'alpha031'
# ,'alpha029'
# ,'alpha025'
# ,'alpha024'
# ,'alpha023'
# ,'alpha022'
# ,'alpha021'
# ,'alpha019'
# ,'alpha017'
# ,'alpha014'
# ,'alpha012'
# ,'alpha011'
# ,'alpha010'
# ,'alpha007'
# ,'alpha004']
# =============================================================================

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
# =============================================================================
#     # 特征重要性
#     features = ['index_close', 'high_rate', 'index_close_diff_1',
#                       'index_close_diff_5', 'index_close_diff_30', 'low_rate',
#                       'turnover_pct', 'turnover', 'atr', 'index_volume',
#                       'index_volume_diff_1', 'index_turnover', 'index_turnover_diff_30',
#                       'index_volume_diff_5', 'date_diff_next', 'index_volume_diff_30',
#                       'index_turnover_diff_5', 'date_week', 'index_turnover_diff_1',
#                       'rsi', 'ad', 'index_turnover_pct', 'alpha041', 'minus_di',
#                       'alpha005', 'index_close_rate', 'plus_di', 'lower_band',
#                       'alpha029', 'volume', 'wr', 'close_diff_30', 'alpha001',
#                       'alpha008', 'adx', 'board_type', 'alpha019', 'alpha011',
#                       'alpha054', 'close_rate', 'alpha047', 'alpha053', 'cci',
#                       'kdj_fastk', 'alpha034', 'alpha083', 'alpha033', 'kdj_fastd',
#                       'alpha025', 'alpha101', 'alpha073', 'alpha040', 'alpha018',
#                       'alpha088', 'close_hist_12_26_9', 'close_acceleration_12_26_9',
#                       'alpha039', 'alpha060', 'alpha077', 'open_rate', 'pb_mrq',
#                       'alpha057', 'close_hist_diff_1', 'chg_rel', 'price_limit_rate',
#                       'close_diff_5', 'alpha037', 'alpha042', 'alpha038', 'alpha052',
#                       'mom', 'alpha035', 'alpha028', 'upper_band', 'alpha031',
#                       'alpha024', 'date_diff_prev', 'alpha004', 'turnover_diff_1',
#                       'alpha084', 'turnover_diff_5', 'pe_ttm', 'alpha014', 'mfi',
#                       'alpha030', 'turnover_pct_slope_12_26_9', 'alpha020', 'alpha002',
#                       'vwap', 'close_slope_12_26_9', 'alpha032', 'alpha036', 'alpha022',
#                       'alpha046', 'alpha045', 'turnover_pct_acceleration_12_26_9',
#                       'turnover_diff_30', 'middle_band', 'volume_diff_1', 'alpha017',
#                       'alpha043', 'alpha009', 'turnover_pct_hist_12_26_9',
#                       'volume_diff_30', 'alpha003', 'alpha078', 'alpha085',
#                       'turnover_hist_diff_1', 'alpha007', 'alpha016', 'alpha023',
#                       'alpha013', 'alpha072', 'alpha049', 'alpha098', 'alpha094',
#                       'alpha010', 'alpha006', 'close_diff_1', 'volume_hist_12_26_9',
#                       'ps_ttm', 'alpha055', 'turnover_slope_12_26_9', 'alpha050',
#                       'volume_diff_5', 'alpha071', 'alpha044',
#                       'volume_acceleration_12_26_9', 'alpha092', 'alpha065',
#                       'turnover_hist_12_26_9']
# =============================================================================
    raw_df[categorical_features] = raw_df[categorical_features].astype('category')
    
    raw_df.sort_values(by='date', ascending=True, inplace=True, ignore_index=True)
    #y_10d_vwap, next_based_index_class
    raw_df[['y_10d_vwap_rate']] = raw_df[['y_10d_vwap']].div(raw_df['close'], axis=0)
    raw_df = raw_df[~raw_df['y_10d_vwap_rate'].isna()]

    # 特征和目标
    X = raw_df.loc[raw_df.date<='2024-12-02', features]  # raw_df[features+['date']]#
    y = raw_df.loc[raw_df.date<='2024-12-02', 'y_10d_vwap_rate']  # raw_df[['next_based_index_class','date']]#
    X_val = raw_df.loc[raw_df.date>'2024-12-03', features]
    y_val = raw_df.loc[raw_df.date>'2024-12-03', 'y_10d_vwap_rate']
    
    # 滚动训练和交叉验证
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # 保存模型在每个滚动窗口的表现
    scores = []
    # 训练模型
    params={'objective': 'regression',
            #'num_class': 3,
            'boosting_type': 'gbdt',#dart, gbdt
            'metric': 'mae',#multi_error,multi_logloss
            'num_leaves': 255,
            'max_depth': 8,
            'learning_rate': 0.12,
            'n_estimators': 1500,#int(200/n_splits),
            'min_gain_to_split':0.1,
            }
    model = lgb.LGBMRegressor(#device='gpu',
                               #gpu_platform_id=0,
                               #gpu_device_id=0,
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

# =============================================================================
#     model.fit(X,
#               y,
#               eval_set= [(X_val, y_val)],
#               eval_metric='multi_logloss',
#               )
# =============================================================================
    for train_index, test_index in tscv.split(X):
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        logger.info(f'{X_train.shape[0]} {train_index.max()}')
        logger.info(f'{X_test.shape[0]} {test_index.min()}')
        
        model.fit(X_train,
                  y_train,
                  eval_set= [(X_test, y_test)],
                  #eval_set= [(X_val, y_val)],
                  #eval_metric='f05_scorer',#f05_scorer,multi_error,multi_logloss
                  #init_model = 'LGBMModel',
                  )
        logger.info(f'模型迭代次数: {model.n_iter_}')
        # 评估
# =============================================================================
#         y_pred = model.predict(X_test)
#         class_report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T
#         class_report_df=class_report_df.head(-3) # del 'accuracy', 'macro avg', 'weighted avg'
#         fbeta = fbeta_score(y_test, y_pred, beta=0.5, average=None)
#         class_report_df['f05-score'] = fbeta
#         print(class_report_df)
# =============================================================================
        #scores.append([fbeta,sum(fbeta)])

        y_val_pred = model.predict(X_test)
        
    logger.info(f'模型迭代次数: {model.n_iter_}')
    y_val_pred = model.predict(X_val)
    result_reg = pd.DataFrame([y_val.values, y_val_pred]).T
    result_reg.columns=['y_real','y_pred']
    result_reg = result_reg.round(4)
    result_reg.to_csv(f'{path}/data/test_reg_5d_vwap.csv', index=False)
    
    # result_reg.loc[result_reg.y_pred>1.05,'y_real'].prod()
    
# =============================================================================
#         y_val_pred = model.predict(X_test)
#         conf_matrix = confusion_matrix(y_test, y_val_pred, labels=[-2, -1, 0, 1, 2])
#         print(conf_matrix)
#         class_report_df = pd.DataFrame(classification_report(y_test, y_val_pred, output_dict=True)).T
#         class_report_df=class_report_df.head(-3) # del 'accuracy', 'macro avg', 'weighted avg'
#         fbeta = fbeta_score(y_test, y_val_pred, beta=0.5, average=None)
#         class_report_df['f05-score'] = fbeta
#         print(class_report_df)
#     #print(scores)
#    
#     # val
#     logger.info(f'模型迭代次数: {model.n_iter_}')
#     y_val_pred = model.predict(X_val)
#     conf_matrix = confusion_matrix(y_val.values, y_val_pred, labels=[-2, -1, 0, 1, 2])
#     print(conf_matrix)
#     class_report_df = pd.DataFrame(classification_report(y_val, y_val_pred, output_dict=True)).T
#     class_report_df=class_report_df.head(-3) # del 'accuracy', 'macro avg', 'weighted avg'
#     fbeta = fbeta_score(y_val.values, y_val_pred, beta=0.5, average=None)
#     class_report_df['f05-score'] = fbeta
#     print(class_report_df)
# 
#     result_class = pd.DataFrame([y_val.values, y_val_pred]).T
#     result_class.columns=['y_real','y_pred']
#     result_class[result_class.y_real==2]
#     result_class[result_class.y_pred==2].sample(n=25, random_state=42).value_counts()
#     print(result_class.head(40))
#     result_class[result_class.y_real==result_class.y_pred]
# =============================================================================
    
    ##按日分布的F05
    
    ##如果只取每日的TOP3，准确率有多少
    
    #importance_weight = model.feature_importance(importance_type='weight')  # 基于增益

# =============================================================================
#     feature_df = pd.DataFrame()
#     feature_df["feature"] = model.feature_name_   
#     feature_df["importance"] = model.feature_importances_
#     feature_df = feature_df.sort_values(by='importance', ascending=False)
#     feature_df.to_csv(f'{path}/data/feature_df.csv', index=False)
#     feature_importance_df = feature_df[feature_df.importance>5].feature.values
# =============================================================================
    
# =============================================================================
# [40]	valid_0's multi_logloss: 0.883311
#    precision    recall  f1-score   support  f05-score
# 0   0.552728  0.459395  0.501758  125661.0   0.531146
# 1   0.623742  0.816335  0.707160  257599.0   0.654630
# 2   0.580034  0.283079  0.380473  113915.0   0.479445
# Did not meet early stopping. Best iteration is:
# [40]	valid_0's multi_logloss: 0.901297
#    precision    recall  f1-score   support  f05-score
# 0   0.610351  0.600159  0.605212  162282.0   0.608285
# 1   0.534507  0.700186  0.606230  189037.0   0.561059
# 2   0.634371  0.391311  0.484041  145856.0   0.564272
# 1491529 497175
# 
# =============================================================================
#fbeta = fbeta_score(y_test, y_pred, beta=0.5, average=None)
#Definition : fit(X: _LGBM_ScikitMatrixLike, y: _LGBM_LabelType, sample_weight: Optional[_LGBM_WeightType]=None, init_score: Optional[_LGBM_InitScoreType]=None, eval_set: Optional[List[_LGBM_ScikitValidSet]]=None, eval_names: Optional[List[str]]=None, eval_sample_weight: Optional[List[_LGBM_WeightType]]=None, eval_class_weight: Optional[List[float]]=None, eval_init_score: Optional[List[_LGBM_InitScoreType]]=None, eval_metric: Optional[_LGBM_ScikitEvalMetricType]=None, feature_name: _LGBM_FeatureNameConfiguration='auto', categorical_feature: _LGBM_CategoricalFeatureConfiguration='auto', callbacks: Optional[List[Callable]]=None, init_model: Optional[Union[str, Path, Booster, LGBMModel]]=None) -> "LGBMClassifier"

