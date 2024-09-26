# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 11:10:25 2024

@author: awei
(backtest_vectorbt_moving_average)
移动平均线
ETF 2019-01-01,2022-01-01三年全量100*100/2均线, 34分钟, 2731676 rows x 52 columns
"""
import os
import itertools

import vectorbt as vbt
import pandas as pd


from __init__ import path
from base import base_log
from backtest.backtest_vectorbt import backtestVectorbt

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = base_log.logger_config_local(f'{path}/log/{log_filename}.log')


class backtestVectorbtMovingAverage(backtestVectorbt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def strategy_params(self, strategy_params_list):
        strategy_params_df = pd.DataFrame(strategy_params_list)
        strategy_params_df['primary_key'] = strategy_params_df['window_fast'].astype(str) + '-' + \
                                            strategy_params_df['window_slow'].astype(str)
        return strategy_params_df
    
    def strategy(self, subtable_df, data):
        window_fast, window_slow = subtable_df[['window_fast', 'window_slow']].values[0]
        
        # Define simple moving average (SMA)
        fast = vbt.MA.run(data, window=window_fast)
        slow = vbt.MA.run(data, window=window_slow)
        
        # Generate buy and sell signals
        entries = fast.ma_crossed_above(slow)
        exits = fast.ma_crossed_below(slow)
        
        entries_exits_t = pd.concat([entries, exits], axis=1, keys=['entries', 'exits']).T
        return entries_exits_t
    
    def ablation_experiment(self, symbols=[],
                            date_start='2020-01-01',
                            date_end='2022-01-01',
                            comparison_experiment=None,
                            if_exists='fail',  # ['fail','replace','append']
                            ):
        # dataset
        portfolio_df = self.dataset(symbols, date_start=date_start, date_end=date_end)
        
        # base
        self.backtest(data=portfolio_df)
        
        # strategy
        self.backtest(portfolio_df, comparison_experiment=comparison_experiment)

        
if __name__ == '__main__':
        # strategy
        params_combinations = [
            {'window_fast': str(window_fast), 'window_slow': str(window_slow)}
            for window_fast, window_slow in itertools.product(list(range(1,100,3)), list(range(1,100,3))) if window_slow > window_fast
        ]
        

        

        
if __name__ == '__main__':
    # strategy
    strategy_params_list = [
        {'window_fast': window_fast, 'window_slow': window_slow}
        for window_fast, window_slow in itertools.product(list(range(1, 100)), list(range(1,100))) if window_slow > window_fast
    ]
    backtest_vectorbt_moving_average = backtestVectorbtMovingAverage(output='database',
                                                                     output_trade_details=False,
                                                                     strategy_params_batch_size=70,
                                                                     portfolio_params={'freq': 'd',
                                                                                       'fees': 0.001,  # 0.1% per trade
                                                                                       'slippage': 0.001,  # 0.1% slippage
                                                                                       'init_cash': 10000},
                                                                     strategy_params_list=strategy_params_list,)
    backtest_vectorbt_moving_average.ablation_experiment(date_start='2019-01-01',
                                                         date_end='2022-01-01',
                                                         comparison_experiment="moving_average_20240913",
                                                         )
    
    

    
