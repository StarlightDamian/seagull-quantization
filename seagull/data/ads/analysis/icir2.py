# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 20:19:33 2025

@author: awei
icir2
apply_func
"""

import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats
from typing import Union, Optional
import warnings
warnings.filterwarnings('ignore')

class VBTFactorAnalyzer:
    def __init__(
        self,
        factor_data: pd.DataFrame,  # 因子值矩阵 (股票 x 时间)
        forward_returns: pd.DataFrame,  # 未来收益率矩阵 (股票 x 时间)
        group_data: Optional[pd.DataFrame] = None  # 分组数据，例如行业、市值分组等
    ):
        """
        初始化分析器
        :param factor_data: 因子值DataFrame，index为股票代码，columns为时间
        :param forward_returns: 未来收益率DataFrame，结构同factor_data
        :param group_data: 分组数据DataFrame，用于分组分析
        """
        self.factor_data = factor_data
        self.forward_returns = forward_returns
        self.group_data = group_data
        self.results = {}

    @staticmethod
    def winsorize(data: Union[pd.Series, pd.DataFrame], n_std: float = 3) -> Union[pd.Series, pd.DataFrame]:
        """
        使用vectorbt进行高效的去极值处理
        """
        mean = data.mean()
        std = data.std()
        return data.clip(lower=mean - n_std * std, upper=mean + n_std * std)

    def neutralize(
        self,
        data: pd.DataFrame,
        neutralizers: pd.DataFrame
    ) -> pd.DataFrame:
        """
        使用vectorbt进行高效的中性化处理
        """
        # 添加常数项
        neutralizers = pd.concat([pd.Series(1, index=neutralizers.index, name='const'), neutralizers], axis=1)
        
        # 使用vbt的并行计算进行中性化
        def neutralize_single(x):
            mask = ~(x.isna() | neutralizers.isna().any(axis=1))
            if mask.sum() > 0:
                reg = np.linalg.lstsq(neutralizers[mask], x[mask], rcond=None)[0]
                residuals = x - neutralizers.dot(reg)
                return residuals
            return x

        return data.apply(neutralize_single)

    @staticmethod
    def standardize(data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """
        使用vectorbt进行高效的标准化处理
        """
        return (data - data.mean()) / data.std()

    def calculate_ic(
        self,
        method: str = 'spearman',
        by_group: bool = False
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        使用vectorbt计算IC值
        :param method: 'spearman' 或 'pearson'
        :param by_group: 是否按组计算IC
        :return: IC值序列或DataFrame（如果by_group=True）
        """
        if not by_group or self.group_data is None:
            ic_func = lambda x, y: stats.spearmanr(x, y)[0] if method == 'spearman' else stats.pearsonr(x, y)[0]
            ic_series = pd.Series(index=self.factor_data.columns)
            
            # 使用vbt的并行计算
            for date in self.factor_data.columns:
                factor = self.factor_data[date].dropna()
                returns = self.forward_returns[date].dropna()
                common_idx = factor.index.intersection(returns.index)
                if len(common_idx) > 0:
                    ic_series[date] = ic_func(factor[common_idx], returns[common_idx])
            
            return ic_series
        else:
            ic_df = pd.DataFrame(index=self.factor_data.columns)
            for group in self.group_data.unique():
                group_stocks = self.group_data[self.group_data == group].index
                group_factor = self.factor_data.loc[group_stocks]
                group_returns = self.forward_returns.loc[group_stocks]
                ic_df[group] = self.calculate_ic(method=method, by_group=False)
            return ic_df

    def calculate_icir(self, ic_series: pd.Series) -> float:
        """
        计算ICIR
        """
        return ic_series.mean() / ic_series.std()

    def analyze(
        self,
        winsorize: bool = True,
        neutralize_data: Optional[pd.DataFrame] = None,
        standardize: bool = True,
        method: str = 'spearman',
        by_group: bool = False
    ):
        """
        完整的因子分析流程
        """
        processed_data = self.factor_data.copy()

        # 数据预处理
        if winsorize:
            processed_data = vbt.apply_func(
                self.winsorize,
                processed_data,
                n_std=3,
                use_ray=True  # 使用Ray进行并行计算
            )

        if neutralize_data is not None:
            processed_data = self.neutralize(processed_data, neutralize_data)

        if standardize:
            processed_data = vbt.apply_func(
                self.standardize,
                processed_data,
                use_ray=True
            )

        # 计算IC和ICIR
        ic_values = self.calculate_ic(method=method, by_group=by_group)
        
        if not by_group:
            icir = self.calculate_icir(ic_values)
            self.results = {
                'IC_series': ic_values,
                'IC_mean': ic_values.mean(),
                'IC_std': ic_values.std(),
                'ICIR': icir,
                'IC_positive_ratio': (ic_values > 0).mean(),
                'IC_decay': self.analyze_ic_decay()
            }
        else:
            self.results = {
                'IC_by_group': ic_values,
                'IC_mean_by_group': ic_values.mean(),
                'IC_std_by_group': ic_values.std(),
                'ICIR_by_group': ic_values.mean() / ic_values.std(),
                'IC_positive_ratio_by_group': (ic_values > 0).mean()
            }

        return self.results

    def analyze_ic_decay(self, max_lag: int = 10) -> pd.Series:
        """
        分析IC衰减
        """
        ic_decay = pd.Series(index=range(1, max_lag + 1))
        
        for lag in range(1, max_lag + 1):
            lagged_returns = self.forward_returns.shift(-lag)
            ic_series = self.calculate_ic(factor_data=self.factor_data, forward_returns=lagged_returns)
            ic_decay[lag] = ic_series.mean()
            
        return ic_decay

# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    stocks = [f'stock_{i}' for i in range(100)]
    
    factor_data = pd.DataFrame(
        np.random.randn(len(stocks), len(dates)),
        index=stocks,
        columns=dates
    )
    
    forward_returns = pd.DataFrame(
        np.random.randn(len(stocks), len(dates)),
        index=stocks,
        columns=dates
    )
    
    # 创建示例行业分组数据
    industries = ['IT', 'Finance', 'Consumer', 'Industry']
    group_data = pd.Series(
        np.random.choice(industries, size=len(stocks)),
        index=stocks
    )
    
    # 初始化分析器
    analyzer = VBTFactorAnalyzer(
        factor_data=factor_data,
        forward_returns=forward_returns,
        group_data=group_data
    )
    
    # 运行分析
    results = analyzer.analyze(
        winsorize=True,
        standardize=True,
        method='spearman',
        by_group=True
    )
    
    # 打印结果
    print("\nAnalysis Results:")
    for key, value in results.items():
        print(f"\n{key}:")
        print(value)
# =============================================================================
# import vectorbt as vbt
# import numpy as np
# import pandas as pd
# from seagull.settings import PATH
# 
# def _apply_full_code(df):
#     return df.pct_change().shift(-1)
# 
# if __name__ == "__main__":
#     raw_df = pd.read_feather(f'{PATH}/data/das_wide_incr_train.feather')
#     
#     alpha_features = ['alpha001', 'alpha002', 'alpha003', 'alpha004', 'alpha005', 'alpha006', 'alpha007', 'alpha008', 'alpha009', 'alpha010', 'alpha011', 'alpha012', 'alpha013', 'alpha014', 'alpha015', 'alpha016', 'alpha017', 'alpha018', 'alpha019', 'alpha020', 'alpha021', 'alpha022', 'alpha023', 'alpha024', 'alpha025', 'alpha026', 'alpha027', 'alpha028', 'alpha029', 'alpha030', 'alpha031', 'alpha032', 'alpha033', 'alpha034', 'alpha035', 'alpha036', 'alpha037', 'alpha038', 'alpha039', 'alpha040', 'alpha041', 'alpha042', 'alpha043', 'alpha044', 'alpha045', 'alpha046', 'alpha047', 'alpha049', 'alpha050', 'alpha051', 'alpha052', 'alpha053', 'alpha054', 'alpha055', 'alpha057', 'alpha060', 'alpha061', 'alpha062', 'alpha064', 'alpha065', 'alpha066', 'alpha068', 'alpha071', 'alpha072', 'alpha073', 'alpha074', 'alpha075', 'alpha077', 'alpha078', 'alpha081', 'alpha083', 'alpha084', 'alpha085', 'alpha086', 'alpha088', 'alpha092', 'alpha094', 'alpha095', 'alpha096', 'alpha098', 'alpha099', 'alpha101']
# 
#     features = alpha_features#ohlc_features +\
#                #fundamental_features +\
#                #label_features +\
#                #macd_features +\
#                #alpha_features #+\
#                #index_features +\
#                #indicators_features
# 
#     factor_df = raw_df[features+['full_code']]
#     
#     # 假设你已经有一个价格数据 df['close'] 和一个因子数据 df['factor']
#     #price_data = pd.read_csv('price_data.csv', index_col='date', parse_dates=True)
#     #factor_data = pd.read_csv('factor_data.csv', index_col='date', parse_dates=True)
# 
#     # 计算因子收益率
#     #factor_returns = factor_df.pct_change().shift(-1)
#     factor_returns = factor_df.groupby('full_code').apply(_apply_full_code)
#     del factor_returns['full_code']
#     # 计算因子与未来收益之间的相关性（IC）
#     factor_returns = factor_returns.dropna(axis=0, how='any')
#     factor_returns = factor_returns.loc[:, factor_returns.notna().any(axis=0)]
# 
#     ic = vbt.factors.factor_ic(factor_df, raw_df['close'])
#     print(f"IC: {ic.mean()}")
# 
#     # 计算IR（信息比率）
#     factor_returns = factor_returns.dropna()
#     ir = factor_returns.mean() / factor_returns.std()
#     print(f"IR: {ir}")
#     
#     
#     factor_df.alpha001
# 
# =============================================================================
