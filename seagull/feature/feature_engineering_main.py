"""
Created on Wed Nov 29 17:07:09 2023

@author: awei
特征工程主程序(feature_engineering_main)


"""
import os
import argparse

import numpy as np
import pandas as pd
from sklearn.utils import Bunch

from __init__ import path
from utils import utils_database, utils_log
from finance import finance_trading_day

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{path}/log/{log_filename}.log')

pd.options.mode.chained_assignment = None
TARGET_NAMES = ['next_low_rate',
                     'next_high_rate',
                     ]  

class FeatureEngineering(finance_trading_day.TradingDayAlignment):
    def __init__(self, target_names=TARGET_NAMES):
        """
        初始化函数，用于登录系统和加载行业分类数据
        :param check:是否检修中间层asset_df
        """
        super().__init__()
        self.target_names = target_names
        
    def feature_merge(self, df):
        """
        特征工程的主要流程，包括指定交易日、创建待预测值、构建数据集
        :param df: 包含日期范围的DataFrame
        :return: 包含数据集的Bunch
        """
        ohlc_features = ['open_rate', 'high_rate', 'low_rate', 'close_rate', 'volume', 'turnover', 'turnover_pct',
                         'price_limit_rate','board_type', 'date_diff_prev', 'date_diff_next', 'date_week','is_limit_down_prev',
                         'is_limit_up_prev']
        fundamental_features = ['chg_rel', 'pe_ttm', 'ps_ttm', 'pcf_ttm', 'pb_mrq']
        # =============================================================================
        #     # 多标签进行onehot， 先置空加快计算
        #     ohlc_df = onehot(ohlc_df, label_df[['full_code', 'label']])
        #     label_features =  [x for x in ohlc_df.columns if ('label_' in x) and ('label_昨日' not in x)]#[]#
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
# =============================================================================
#         flow_features = ['loge_main_inflow', 'loge_ultra_large_inflow',
#                'loge_large_inflow', 'loge_medium_inflow', 'loge_small_inflow','loge_main_small_net_inflow',
#                'main_inflow_slope_12_26_9', 'ultra_large_inflow_slope_12_26_9',
#                'large_inflow_slope_12_26_9', 'medium_inflow_slope_12_26_9',
#                'small_inflow_slope_12_26_9', 'main_inflow_acceleration_12_26_9',
#                'ultra_large_inflow_acceleration_12_26_9',
#                'large_inflow_acceleration_12_26_9',
#                'medium_inflow_acceleration_12_26_9',
#                'small_inflow_acceleration_12_26_9', 'main_inflow_hist_12_26_9',
#                'ultra_large_inflow_hist_12_26_9', 'large_inflow_hist_12_26_9',
#                'medium_inflow_hist_12_26_9', 'small_inflow_hist_12_26_9',
#                'main_inflow_diff_1', 'main_inflow_diff_5', 'main_inflow_diff_30',
#                'ultra_large_inflow_diff_1', 'ultra_large_inflow_diff_5',
#                'ultra_large_inflow_diff_30', 'large_inflow_diff_1',
#                'large_inflow_diff_5', 'large_inflow_diff_30', 'medium_inflow_diff_1',
#                'medium_inflow_diff_5', 'medium_inflow_diff_30', 'small_inflow_diff_1',
#                'small_inflow_diff_5', 'small_inflow_diff_30',
#                'main_inflow_hist_diff_1', 'ultra_large_inflow_hist_diff_1',
#                'large_inflow_hist_diff_1', 'medium_inflow_hist_diff_1',
#                'small_inflow_hist_diff_1',
#                'main_small_net_inflow_slope_12_26_9',
#                'main_small_net_inflow_acceleration_12_26_9',
#                'main_small_net_inflow_hist_12_26_9',
#                'main_small_net_inflow_diff_1',
#                'main_small_net_inflow_diff_5',
#                'main_small_net_inflow_diff_30',
#                'main_small_net_inflow_hist_diff_1']
# =============================================================================
        alpha_features = ['alpha001', 'alpha002', 'alpha003', 'alpha004', 'alpha005', 'alpha006', 'alpha007', 'alpha008', 'alpha009', 'alpha010', 'alpha011', 'alpha012', 'alpha013', 'alpha014', 'alpha015', 'alpha016', 'alpha017', 'alpha018', 'alpha019', 'alpha020', 'alpha021', 'alpha022', 'alpha023', 'alpha024', 'alpha025', 'alpha026', 'alpha027', 'alpha028', 'alpha029', 'alpha030', 'alpha031', 'alpha032', 'alpha033', 'alpha034', 'alpha035', 'alpha036', 'alpha037', 'alpha038', 'alpha039', 'alpha040', 'alpha041', 'alpha042', 'alpha043', 'alpha044', 'alpha045', 'alpha046', 'alpha047', 'alpha049', 'alpha050', 'alpha051', 'alpha052', 'alpha053', 'alpha054', 'alpha055', 'alpha057', 'alpha060', 'alpha061', 'alpha062', 'alpha064', 'alpha065', 'alpha066', 'alpha068', 'alpha071', 'alpha072', 'alpha073', 'alpha074', 'alpha075', 'alpha077', 'alpha078', 'alpha081', 'alpha083', 'alpha084', 'alpha085', 'alpha086', 'alpha088', 'alpha092', 'alpha094', 'alpha095', 'alpha096', 'alpha098', 'alpha099', 'alpha101']
        index_features = ['index_close_diff_1', 'index_close_diff_5', 'index_close_diff_30', 'index_volume_diff_1','index_volume_diff_5',
                          'index_volume_diff_30', 'index_turnover_diff_1', 'index_turnover_diff_5', 'index_turnover_diff_30','index_close_rate',
                          'index_close','index_volume','index_turnover','index_turnover_pct']
        indicators_features = ['rsi','cci', 'wr', 'vwap', 'ad', 'mom', 'atr', 'adx', 'plus_di', 'minus_di', 'mfi', 'upper_band', 'middle_band', 'lower_band', 'kdj_fastk','kdj_fastd']
        
        feature_names = ['primary_key'] +\
                ohlc_features +\
                   fundamental_features +\
                   label_features +\
                   macd_features +\
                   alpha_features +\
                   index_features +\
                   indicators_features
                   # flow_features +\
# =============================================================================
#         # 删除非训练字段
#         feature_names = df.columns.tolist()board_type
#         columns_to_drop = ['date', 'time', 'freq', 'market_code', 'full_code', 'asset_code',
#                            'code_name', 'adj_type','st_status', 'prev_date',
#                            'next_low', 'next_high', 'next_open', 'next_close',
#                            'prev_macro_value_traded','board_type','price_limit_rate',
#                            'board_primary_key','insert_timestamp'] + self.target_names
#         feature_names = list(set(feature_names) - set(columns_to_drop))
#         feature_names = [x for x in feature_names if '_real' not in x]
# =============================================================================

        constant_features = df.columns[df.nunique() == 1]  # 筛选出所有值相同的列
        logger.warning(f'相同值特征: {constant_features}')  # ['alpha019', 'alpha027', 'alpha039']
        df = df.loc[:, df.nunique() > 1]
        
        # 特征选择
        categorical_features = ['date_week']#'full_code',
        numeric_features = list(set(feature_names)-set(constant_features)-set(categorical_features)-
                                set(['alpha061','alpha062','alpha064','alpha065','alpha068','alpha074','alpha075',
                                     'alpha081','alpha086','alpha095','alpha099']))
# =============================================================================
#     wide_df.alpha095.unique()
#     Out[8]: array([False, True, nan], dtype=object)
# =============================================================================
# =============================================================================
#     Fields with bad pandas dtypes: alpha062: object, alpha064: object, alpha065: object, alpha074: object, alpha081: object, alpha099: object
# # =============================================================================
# =============================================================================
# ValueError: pandas dtypes must be int, float or bool.
# Fields with bad pandas dtypes: alpha061: object, alpha075: object, alpha095: object
# =============================================================================
        return df, numeric_features, categorical_features
    
    def build_dataset(self, asset_df, numeric_features, categorical_features):
        ## 构建数据集
        # 输出有序标签
        numeric_features = sorted(numeric_features)
        categorical_features = sorted(categorical_features)
        # print(f'feature_names_engineering:\n {feature_names}')
        #print('self.target_names',self.target_names)
        date_range_dict = {'data': np.array(asset_df[numeric_features + categorical_features].to_records(index=False)),  # 不使用 feature_df.values,使用结构化数组保存每一列的类型
                           'numeric_features': numeric_features,
                           'categorical_features': categorical_features,
                           'target': asset_df[self.target_names].values,  # 机器学习预测值
                           'target_names': [self.target_names],
                           }
        bunch = Bunch(**date_range_dict)
        return bunch
    
    def feature_engineering_pipeline(self, asset_df):
        """
        特征工程的主要流程，包括指定交易日、创建待预测值、构建数据集
        :param asset_df: 包含日期范围的DataFrame
        :return: 包含数据集的Bunch
        """
        # 构建数据集
        logger.info(f'asset_shape_input: {asset_df.shape}')
        asset_df, numeric_features, categorical_features = self.feature_merge(asset_df)
        logger.info(f'asset_shape_output: {asset_df.shape}')
        logger.info(f'numeric_features: {numeric_features}')
        logger.info(f'categorical_features: {categorical_features}')
        bunch = self.build_dataset(asset_df, numeric_features, categorical_features)
        return bunch
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='2023-09-20', help='When to start feature engineering')
    parser.add_argument('--date_end', type=str, default='2023-12-27', help='End time for feature engineering')
    args = parser.parse_args()
    
    logger.info(f"""task: feature_engineering
                    date_start: {args.date_start}
                    date_end: {args.date_end}""")
    
    # 获取日期段数据
    with utils_database.engine_conn('postgre') as conn:
        asset_df = pd.read_sql(f"SELECT * FROM dwd_ohlc_incr_stock_daily WHERE date >= '{args.date_start}' AND date < '{args.date_end}'", con=conn.engine)
    
    feature_engineering = FeatureEngineering()
    feature_df, numeric_features, categorical_features = feature_engineering.feature_merge(asset_df)
    # date_range_bunch = feature_engineering.pipeline(asset_df)
    
    


