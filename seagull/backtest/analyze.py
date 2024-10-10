# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:50:06 2024

@author: awei
(backtest_analyze)
"""
import os
#import functools

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from __init__ import path
from utils import utils_log, utils_database

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{path}/log/{log_filename}.log')

pd.set_option('display.max_rows', 20)  
pd.set_option('display.max_columns', 20)  


class backtestAnalyze:
    def __init__(self):
        # entries.sum().sum()=9843,  exits.sum().sum()=9728
        ...
        
    def comparison_base_strategy(self, base_df, strategy_df, column='score'):
        # 按ETF分组计分
        base_group_df = base_df.groupby('full_code').apply(lambda x: x[column].mean())
        strategy_group_df = strategy_df.groupby('full_code').apply(lambda x: x[column].mean()) # 取均值，也可以最大值
        portfolio_df = pd.concat([base_group_df, strategy_group_df], axis=1, keys=['base','strategy'])  
        
        portfolio_df = portfolio_df[~portfolio_df.strategy.isnull()]
        portfolio_df['strategy_better'] = portfolio_df['strategy'] - portfolio_df['base']
        portfolio_df = portfolio_df.round(3)
        portfolio_df = portfolio_df.sort_values(by='strategy_better', ascending=False)
        
        # 新增一行均值
        portfolio_df.loc['mean'] = portfolio_df.mean()
        return portfolio_df
        
    def rank_personal(self, bacetest_df):
        bacetest_df = bacetest_df.sort_values(by='score', ascending=False)
        rank_personal_df = bacetest_df[['full_code', 'comparison_experiment','window', 'bar_num', 'ann_return','max_dd','score']]
        return rank_personal_df
    
    def plt(self):
        for idx, strategy_params in enumerate(params_combinations):
            strategy_df_1 = strategy_effective_df[(strategy_effective_df.fast_ma==strategy_params['window_fast'])&
                                                  (strategy_effective_df.slow_ma==strategy_params['window_slow'])]
            if not strategy_df_1.empty:
                merge_df_1 = pd.merge(strategy_df_1, base_df[['symbol','score']], on='symbol', suffixes=('', '_base'))
                score = merge_df_1.score.mean() - merge_df_1.score_base.mean()
                params_combinations[idx]['score'] = score
            else:
                params_combinations[idx]['score'] = 0
        
        # 将数据转换为 DataFrame
        df = pd.DataFrame(params_combinations)
        
        # 创建空矩阵，大小为 5x5
        matrix = np.zeros((len(df.window_fast.unique()),#df.window_fast.astype(int).max(),
                           len(df.window_fast.unique())#df.window_slow.astype(int).max()
                           ))
        
        # 填充矩阵数据，将 'window_fast' 和 'window_slow' 的索引值减 1 作为矩阵的索引
        for row in params_combinations:
            i = int(row['window_fast']) - 1
            j = int(row['window_slow']) - 1
            matrix[i, j] = row['score']
        
        # 设置绘图大小
        plt.figure(figsize=(8, 6))
        
        # 绘制热力图，使用 'RdYlGn' 颜色映射，显示每个格子中的数值
        sns.heatmap(matrix,
                    annot=True,
                    cmap='RdYlGn_r',#,RdYlGn
                    linewidths=0.5,
                    fmt=".2f")
        # 设置轴标签
        plt.xlabel('window_slow')
        plt.ylabel('window_fast')
        
        # 显示图形
        plt.show()
        
    def output_csv(self):
        """
        ads_info_incr_bacetest.columns = ['start', 'end', 'period', 'start_value', 'end_value', 'total_return',
               'benchmark_return', 'max_gross_exposure', 'total_fees_paid', 'max_dd',
               'max_dd_duration', 'total_trades', 'total_closed_trades',
               'total_open_trades', 'open_trade_pnl', 'win_rate', 'best_trade',
               'worst_trade', 'avg_winning_trade', 'avg_losing_trade',
               'avg_winning_trade_duration', 'avg_losing_trade_duration',
               'profit_factor', 'expectancy', 'sharpe_ratio', 'calmar_ratio',
               'omega_ratio', 'sortino_ratio', 'ann_return', 'ann_volatility', 'skew',
               'kurtosis', 'tail_ratio', 'common_sense_ratio', 'value_at_risk',
               'alpha', 'beta', 'profit_loss_ratio', 'freq_stats', 'score', 'symbol',
               'fast_ma', 'slow_ma', 'comparison_experiment', 'fees', 'slippage',
               'freq_portfolio', 'insert_timestamp']
        """
        strategy_portfolio_d = strategy_portfolio_d[['ann_return','max_dd','sharpe_ratio','sortino_ratio','win_rate','profit_loss_ratio','score']]
        strategy_portfolio_d_cn = strategy_portfolio_d.rename(columns={'ann_return': '策略_年化收益 [%]',
                                                                    'max_dd': '策略_最大回撤 [%]',
                                                                    'sharpe_ratio': '策略_夏普比',
                                                                    'sortino_ratio': '策略_sortino风险比',
                                                                    'win_rate': '策略_胜率',
                                                                    'profit_loss_ratio': '策略_盈亏比',
                                                                    'score': '策略_总分',
                                                                    })
        base_portfolio_d[['ann_return','max_dd','sharpe_ratio','sortino_ratio','win_rate','profit_loss_ratio','score']]
        base_portfolio_d_cn = base_portfolio_d.rename(columns={'ann_return': '基准_年化收益 [%]',
                                                            'max_dd': '基准_最大回撤 [%]',
                                                            'sharpe_ratio': '基准_夏普比',
                                                            'sortino_ratio': '基准_sortino风险比',
                                                            'win_rate': '基准_胜率',
                                                            'profit_loss_ratio': '基准_盈亏比',
                                                            'score': '基准_总分',
                                                            })
        
        strategy_portfolio_d_cn['代码'] = base_portfolio_d.index
        base_portfolio_d_cn['代码'] = base_portfolio_d.index
        portfolio_d = pd.merge(strategy_portfolio_d_cn,base_portfolio_d_cn, on='代码')
        portfolio_d['时频'] = '日线'
        portfolio_d = portfolio_d[['代码','时频','基准_年化收益 [%]', '策略_年化收益 [%]', '基准_最大回撤 [%]', '策略_最大回撤 [%]',
                                   '基准_夏普比', '策略_夏普比', '基准_sortino风险比', '策略_sortino风险比',
                                   '基准_胜率','策略_胜率', '基准_盈亏比', '策略_盈亏比','基准_总分','策略_总分']]
        portfolio_d.to_csv(f'{path}/data/portfolio_d.csv', index=False)
        


    
    def pipeline(self, comparison_experiment):
        ## dataset
        with utils_database.engine_conn('postgre') as conn:
            bacetest_raw_df = pd.read_sql(f"select * from ads_info_incr_bacetest where comparison_experiment in ('{comparison_experiment}', 'base')", con=conn.engine)#ads_info_incr_bacetest_signal
        bacetest_df = bacetest_raw_df.drop_duplicates('primary_key', keep='first') # 去重
        bacetest_df = bacetest_df[~bacetest_df.score.isnull()]
        
        bacetest_df['window'] = bacetest_df.fast_ma.astype(str) + '-' + \
                                bacetest_df.slow_ma.astype(str) + '-' + \
                                bacetest_df.signal_ma.astype(str)
        
        # strategy
        strategy_df = bacetest_df[bacetest_df.comparison_experiment==comparison_experiment]
        strategy_effective_df = strategy_df[strategy_df.bar_num/strategy_df.period>0.6]  # effective strategy df
        
        # base
        start, end = strategy_df[['start','end']].values[0]
        base_df = bacetest_df[(bacetest_df.comparison_experiment=='base')&
                              (bacetest_df.start==start)&
                              (bacetest_df.end==end)]
        
        ## analyze
        # chiropractic
        score_mean_base = base_df.score.mean() # 34.279
        score_mean_strategy_effective = strategy_effective_df.score.mean()  # 27.553
        logger.info(f'score_mean_base: {score_mean_base:.3f}')
        logger.info(f'score_mean_strategy_effective: {score_mean_strategy_effective:.3f}')
        
        # import
        # base_df.sort_values(by='score',ascending=False)[['full_code','score','ann_return','max_dd','bar_num']]
        #strategy_effective_df = strategy_effective_df[strategy_effective_df.full_code.isin(symbols)].groupby('remark').apply(lambda x: x.score.mean()).sort_values(ascending=False)
        #strategy_effective_df[strategy_effective_df.full_code.isin(symbols)].groupby('remark').apply(lambda x: x.score.mean()).sort_values(ascending=False) #11-26    40.961417
        #base_df[base_df.full_code.isin(symbols)].score.mean() #42.088
        
        # group-strategy, 按策略分组计分
        #strategy_effective_df.groupby('window').apply(lambda x: x.score.mean()).sort_values(ascending=False).round(3)
        strategy_rank_df = bacetest_df.groupby('window').apply(lambda x: x.score.mean()).sort_values(ascending=False).round(3)
        logger.info(f'strategy_rank: \n{strategy_rank_df}')
        
        # group-portfolio, 按ETF分组计分
        rank_portfolio_df = bacetest_df.groupby('full_code').apply(lambda x: x.score.mean()).sort_values(ascending=False).round(3)
        logger.info(f'rank_portfolio: \n{rank_portfolio_df}')
        
        # group-time, 按时间分组统计
        pass
        
        # baseline
        baseline_df = base_df[base_df.full_code=='SH.510300'].score.values[0]  # 48.932
        baseline_strategy_df = strategy_df[strategy_df.full_code=='SH.510300'].score.max()  # 22.496
        logger.info(f'baseline: {baseline_df}')
        logger.info(f'baseline_strategy: {baseline_strategy_df}')
        
        # comparison-group-score
        comparison_portfolio_df = self.comparison_base_strategy(base_df, strategy_df)
        logger.info(f'comparison_portfolio: \n{comparison_portfolio_df}')
        
        # rank-personal
        rank_personal_df = self.rank_personal(bacetest_df)
        logger.info(f'rank_personal: \n{rank_personal_df}')
        return bacetest_df, base_df, strategy_df
    
    
if __name__ == '__main__':
    comparison_experiment='macd_diff_20241010_6'
    #macd_20240925,macd_diff_20241010,macd_diff_20241010_2
    backtest_analyze = backtestAnalyze()
    bacetest_df, base_df, strategy_df = backtest_analyze.pipeline(comparison_experiment=comparison_experiment)
    portfolio_df = backtest_analyze.comparison_base_strategy(base_df, strategy_df)
    

    