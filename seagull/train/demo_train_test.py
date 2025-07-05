# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 21:04:12 2024

@author: awei
demo_train_test
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from seagull.settings import PATH
from seagull.utils import utils_database


def dataset_1(subtable):
    subtable[['prev_close']] = subtable[['close']].shift(1)
    subtable[['high_pct', 'low_pct', 'open_pct', 'close_pct', 'avg_price_pct']] = subtable[
        ['high', 'low', 'open', 'close', 'avg_price']].div(subtable['prev_close'], axis=0)
    subtable[['next_high_pct', 'next_low_pct', 'next_close_pct']] = subtable[
        ['high_pct', 'low_pct', 'close_pct']].shift(-1)
    subtable = subtable.head(-1)
    return subtable


def dataset(date_start, date_end):
    # with utils_database.engine_conn("POSTGRES") as conn:
    #    df = pd.read_sql(f"SELECT * FROM dwd_freq_incr_stock_daily where date between '{date_start}' and '{date_end}'",
    #                           con=conn.engine)
    df = pd.read_csv(f'{PATH}/_file/test_dwd_freq_incr_stock_daily.csv', low_memory=False)
    # =============================================================================
    #     df = df[['primary_key', 'date', 'time', 'board_type', 'full_code', 'asset_code',
    #            'market_code', 'code_name', 'price_limit_pct', 'open', 'high', 'low',
    #            'close', 'prev_close', 'volume', 'value_traded', 'turnover',
    #            'trade_status', 'chg_pct', 'pe_ttm', 'ps_ttm', 'pcf_ttm', 'pb_mrq',
    #            'st_status', 'insert_timestamp', 'amplitude', 'price_chg', 'freq',
    #            'adj_type', 'board_primary_key', 'avg_price']]
    # =============================================================================
    df = df.groupby('full_code').apply(dataset_1)  # .reset_index(drop=True)

    feature_name = ['high', 'low', 'open', 'close', 'volume',
                    'value_traded', 'turnover', 'chg_pct', 'pe_ttm',
                    'ps_ttm', 'pcf_ttm', 'pb_mrq', 'amplitude', 'price_chg']
    x = df[feature_name]
    y = df['next_high_pct']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test


def custom_loss(y_pred, dataset):
    y_true = dataset.get_label()
    delta = y_true - y_pred
    alpha = 1.2  # Huber 损失的阈值参数

    # 梯度和 Hessian
    grad = np.where(np.abs(delta) <= alpha, -delta, -delta)
    hess = np.where(np.abs(delta) <= alpha, 0.9, 0.9)

    return grad, hess


if __name__ == '__main__':
    with utils_database.engine_conn("POSTGRES") as conn:
        index_df = pd.read_sql("dwd_part_full_index_base", con=conn.engine)
        index_df = index_df[index_df["index"] == 'hs300']
    # =============================================================================
    #     data = pd.read_csv(f'{PATH}/_file/test_603893.csv')
    #     #data['high'] = data['high'] / data['close']
    #     columns_to_divide = ['high', 'low', 'open', 'close']
    #     data[columns_to_divide] = data[columns_to_divide].div(data['preclose'], axis=0)
    #
    #     data[['next_high', 'next_low']] = data[['high', 'low']].shift(-1)
    #     data['next_close_real'] = data['close'].shift(-1)
    #     data = data.head(-1)
    #     feature_name=['open', 'high', 'low', 'close', 'volume',
    #            'amount', 'adjustflag', 'turn', 'tradestatus', 'pctChg', 'peTTM',
    #            'psTTM', 'pcfNcfTTM', 'pbMRQ', 'isST']#,'next_close_real'
    #     x = data[feature_name]
    #     #y = data[['next_high', 'next_low']]
    #     y = data['next_high']
    #     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # =============================================================================

    date_start = '2020-01-01'
    date_end = '2021-01-01'
    df = pd.read_csv(f'{PATH}/_file/test_dwd_freq_incr_stock_daily.csv', low_memory=False)
    df = df[df.full_code.isin(index_df.full_code)]  # 603893.sh, 002230.sz,
    # df = df.groupby('full_code').apply(dataset_1)#.reset_index(drop=True)
    df[['prev_close']] = df[['close']].shift(1)
    df[['high_pct', 'low_pct', 'open_pct', 'close_pct', 'avg_price_pct']] = df[
        ['high', 'low', 'open', 'close', 'avg_price']].div(df['prev_close'], axis=0)
    df[['next_high_pct', 'next_low_pct', 'next_close_pct']] = df[['high_pct', 'low_pct', 'close_pct']].shift(-1)
    df = df.head(-1)
    feature_name = ['open', 'high', 'low', 'close', 'volume',
                    'value_traded', 'turnover', 'chg_pct', 'pe_ttm',
                    'ps_ttm', 'pcf_ttm', 'pb_mrq']  # ,'next_close_real'
    # =============================================================================
    #     feature_name=['high_pct', 'low_pct', 'open_pct', 'close_pct', 'volume',
    #            'value_traded', 'turnover', 'chg_pct', 'pe_ttm',
    #            'ps_ttm', 'pcf_ttm', 'pb_mrq', 'amplitude', 'price_chg']
    # =============================================================================
    # one_hot_encoder = OneHotEncoder(sparse_output=False)
    x = df[feature_name].reset_index(drop=True)  # .head(50000)
    y = df['next_high_pct'].reset_index(drop=True)  # .head(50000)
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # x = scaler.fit_transform(x)
    # x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # x_train, x_test, y_train, y_test = dataset(date_start, date_end)

    # 创建 Dataset 时包含所有特征
    lgb_train = lgb.Dataset(x_train,
                            label=y_train,
                            # feature_name=feature_name,
                            free_raw_data=False,
                            params={'data_type': 'train'},
                            )

    lgb_valid = lgb.Dataset(x_test,
                            label=y_test,
                            reference=lgb_train,
                            # feature_name=feature_name,
                            free_raw_data=False,
                            params={'data_type': 'valid'},
                            )
    # params={'objective': 'rmse'}
    # =============================================================================
    #     params = {
    #     'objective': 'huber',#'huber',  # 直接使用LightGBM内置的Huber损失
    #     'alpha': 1.2,  # 可以调整Huber损失的阈值参数
    #     'metric': 'None',
    #     'learning_rate': 0.05,
    #     'num_leaves': 62,
    #     'min_data_in_leaf': 20,
    #     'max_depth': -1,
    # }
    # =============================================================================
    # =============================================================================
    #     params = {
    #     'objective': custom_huber_loss,#'huber',  # 直接使用LightGBM内置的Huber损失
    #     'alpha': 1.2,  # 可以调整Huber损失的阈值参数
    #     'metric': 'None',
    #     'learning_rate': 0.05,
    #     'num_leaves': 62,
    #     'min_data_in_leaf': 20,
    #     'max_depth': -1,
    # }
    # =============================================================================
    params = {
        'task': 'train',
        'boosting': 'gbdt',
        'objective': 'regression',
        'num_leaves': 37,  # 37,96 决策树上的叶子节点的数量，控制树的复杂度
        'learning_rate': 0.01,  # 0.05,0.1
        'metric': ['mse'],  # 模型通过mae进行优化, root_mean_squared_error进行评估。, 'root_mean_squared_error',mae
        # w×RMSE+(1−w)×MAE
        'verbose': -1,  # 控制输出信息的详细程度，-1 表示不输出任何信息
        'max_depth': 6,
        'n_estimators': 100,
        # 'early_stopping_round':50,
        'min_child_sample': 40,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'max_bin': 63
    }
    # =============================================================================
    #     params = {'objective': 'rmse',#'huber',  # 直接使用LightGBM内置的Huber损失
    #               'metric': 'None',
    #               'learning_rate': 0.05,
    #               'num_leaves': 62,
    #               'min_data_in_leaf': 20,
    #               'max_depth': -1,
    #               }
    # =============================================================================
    model = lgb.LGBMRegressor(**params)
    model.fit(x_train, y_train)

    # =============================================================================
    #     model = lgb.train(
    #             params=params,
    #             train_set=lgb_train,
    #             valid_sets=[lgb_train, lgb_valid],
    #             num_boost_round=100
    #         )
    # =============================================================================
    y_pred = model.predict(x_test)
    result = pd.DataFrame([y_test.values, y_pred]).T
    result.columns = ['y_test', 'y_pred']
    result['next_high_bool'] = np.where(result['y_test'] >= result['y_pred'], 1, None)
    result = result.round(4)
    print(result)
    result.to_csv(f'{PATH}/_file/test_result_xgboost.csv', index=False)

    result_bool = result[result.next_high_bool == 1]
    y_test_mean, y_pred_mean, next_high_bool = result_bool.mean()
    print(result_bool.mean(), result_bool.shape[0], '/', result.shape[0])

    reward = (y_pred_mean - 1) * (result_bool.shape[0])
    reward_all = (y_test_mean - 1) * (result.shape[0])
    reward_pct = reward / reward_all
    print(f'reward_pct: {reward_pct:.4f}')
    # sz002230 = df[df.full_code=='002230.sz']
