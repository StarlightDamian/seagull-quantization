# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 16:17:57 2024

@author: awei
backtest_vectorbt

查看stats属性：https://github.com/polakowo/vectorbt/blob/master/vectorbt/portfolio/base.py
查看returns_stats属性
https://github.com/polakowo/vectorbt/blob/417b3aa52af182ab2b60e1ba7177f98e6bb1c0cb/vectorbt/portfolio/base.py#L4165
vectorbt的数据获取部分需要国外VPN

calmar_ratio = ann_return / max_dd
from_signals

目前问题：
1，一个批次的横向比较数据，未初始价格被第一次有效价格补全，导致开始时间、结束时间为整个批次的开始 / 结束时间。因此建议用来筛选最佳策略，而不适合用来定量分析各只详细情况

"""
import os
import multiprocessing

import vectorbt as vbt
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#%matplotlib inline
from numba import config, set_num_threads

from seagull.settings import PATH
from seagull.utils import utils_log, utils_character, utils_database, utils_data
from data.dwd import dwdData

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')

num_threads = 8  # for example, if you have an 8-core CPU
config.NUMBA_NUM_THREADS = num_threads
set_num_threads(num_threads)

class backtestVectorbt:
    def __init__(self, output='database',
                       use_multiprocess=False,
                       output_trade_details=False,
                       strategy_params_batch_size=20,
                       portfolio_params=None, # 投资组合参数
                       strategy_params_list=[{'window_fast': 10, 'window_slow': 50},
                                             {'window_fast': 10, 'window_slow': 40}],
                       ):
        self.output_database_filename = 'ads_info_incr_backtest'
        self.use_multiprocess=use_multiprocess
        self.strategy_params_df = self.strategy_params(strategy_params_list)
        self.strategy_params_batch_size = strategy_params_batch_size
        self.output = output
        self.output_trade_details = output_trade_details
        self.stats_metrics_dict={"Start": "start",
                                "End": "end",
                                "Period": "period",
                                "Start Value": "start_value",
                                "End Value": "end_value",
                                "Total Return [%]": "total_return",
                                "Benchmark Return [%]": "benchmark_return",
                                "Max Gross Exposure [%]": "max_gross_exposure",
                                "Total Fees Paid": "total_fees_paid",
                                "Max Drawdown [%]": "max_dd",
                                "Max Drawdown Duration": "max_dd_duration",
                                "Total Trades": "total_trades",
                                "Total Closed Trades": "total_closed_trades",
                                "Total Open Trades": "total_open_trades",
                                "Open Trade PnL": "open_trade_pnl",
                                "Win Rate [%]": "win_rate",
                                "Best Trade [%]": "best_trade",
                                "Worst Trade [%]": "worst_trade",
                                "Avg Winning Trade [%]": "avg_winning_trade",
                                "Avg Losing Trade [%]": "avg_losing_trade",
                                "Avg Winning Trade Duration": "avg_winning_trade_duration",
                                "Avg Losing Trade Duration": "avg_losing_trade_duration",
                                "Profit Factor": "profit_factor",
                                "Expectancy": "expectancy",
                                "Sharpe Ratio": "sharpe_ratio",
                                "Calmar Ratio": "calmar_ratio",
                                "Omega Ratio": "omega_ratio",
                                "Sortino Ratio": "sortino_ratio",
                                }
        
        # rename see https://github.com/polakowo/vectorbt/blob/417b3aa52af182ab2b60e1ba7177f98e6bb1c0cb/vectorbt/returns/accessors.py#L998
        # annualized_return= (total_return+1) **(365 / len(data)) - 1
        self.returns_stats_metrics_dict={#'Start': "start",
                                          #'End': "end",
                                          #'Period': "period",
                                          #'Total Return [%]': "total_return",
                                          #'Benchmark Return [%]': "benchmark_return",
                                          "Annualized Return [%]": "ann_return",
                                          "Annualized Volatility [%]": "ann_volatility",
                                          #'Max Drawdown [%]': "max_dd",
                                          #'Max Drawdown Duration': "max_dd_duration",
                                          #'Sharpe Ratio': "sharpe_ratio",
                                          #'Calmar Ratio': "calmar_ratio",
                                          #'Omega Ratio': "omega_ratio",
                                          #'Sortino Ratio': "sortino_ratio",
                                          'Skew': 'skew',
                                          'Kurtosis': 'kurtosis',
                                          'Tail Ratio': 'tail_ratio',
                                          'Common Sense Ratio': 'common_sense_ratio',
                                          'Value at Risk': 'value_at_risk',
                                          'Alpha': 'alpha',
                                          'Beta': 'beta'}
        self.portfolio_params = {'freq': 'd',
                                 'fees': 0.001,
                                 'slippage': 0.001,
                                 'init_cash': 10000}
        if portfolio_params:
            self.portfolio_params.update(portfolio_params)
        
        self.data_dwd = dwdData()
        
    def score(self, df, weights={"ann_return": 0.70,
                                "max_dd": 0.25,
                                "sharpe_ratio": 0.00,
                                "sortino_ratio": 0.00,
                                "calmar_ratio": 0.00,
                                "alpha": 0.00,
                                'ann_volatility': 0.05,
                                "omega_ratio": 0.00,
                                "beta": 0.00,
                                }):
        
        # Normalize values (assuming higher is better for most metrics)
        norm_ann_return = df['ann_return'] / 100
        norm_max_dd = 1 - (df['max_dd'] / 100)  # Penalize drawdown
        norm_sharpe = df['sharpe_ratio'] / 2  # Arbitrary normalization factor
        norm_calmar = df['calmar_ratio'] / 5
        norm_sortino = df['sortino_ratio'] / 5
        norm_ann_volatility = 1 - (df['ann_volatility'] / 100)
        norm_omega = df['omega_ratio'] / 5
        norm_alpha = df['alpha']
        norm_beta = 1 - df['beta']  # Lower beta is usually better
        
        # Compute weighted score
        score = (weights['ann_return'] * norm_ann_return +
                 weights['max_dd'] * norm_max_dd +
                 weights['sharpe_ratio'] * norm_sharpe +
                 weights['calmar_ratio'] * norm_calmar +
                 weights['sortino_ratio'] * norm_sortino +
                 weights['omega_ratio'] * norm_omega +
                 weights['alpha'] * norm_alpha +
                 weights['ann_volatility'] * norm_ann_volatility +
                 weights['beta'] * norm_beta
                 ) * 100
    
        return score
    
    def profit_loss_ratio(self, portfolio):
        returns_df = portfolio.returns()
        positive_returns = returns_df[returns_df > 0].sum()
        negative_returns = abs(returns_df[returns_df < 0].sum())
        result = positive_returns / negative_returns.where(negative_returns != 0, np.inf)  # To handle division by zero
        return result
        
    def portfolio_stats(self, portfolio, freq='d'):
        # Focus: This function provides broader performance metrics about the entire portfolio, covering not only the returns but also information about the overall portfolio's activity and performance.
        try:
            backtest_df = portfolio.stats(
                                    metrics=self.stats_metrics_dict.values(),  # ['sharpe_ratio', 'max_dd']
                                    settings=dict(freq=freq,
                                                  year_freq='243 days'),  # freq in ['d','30d','365d']
                                    #group_by=False,
                                    agg_func=None)
            backtest_df = backtest_df.rename(columns=self.stats_metrics_dict)
            return backtest_df
        except OverflowError:
            logger.error('int too big to convert, The backtest freq date is greater than the data date.')
            return None

    def portfolio_returns_stats(self, portfolio, freq='d'):
        # This function focuses specifically on returns and statistical metrics related to the return series.
        try:
            # https://github.com/polakowo/vectorbt/blob/54cbe7c5bff332b510d1075c5cf11d006c1b1846/vectorbt/portfolio/base.py#L2022
            backtest_df = portfolio.returns_stats(metrics=self.returns_stats_metrics_dict.values(),
                                                          settings=dict(freq=freq),
                                                          year_freq='243 days',
                                                          agg_func=None)
            backtest_df = backtest_df.rename(columns=self.returns_stats_metrics_dict)
            return backtest_df
        except OverflowError:
            logger.error('int too big to convert, The backtest freq date is greater than the data date.')
            return None

    def portfolio_trades_stats(self, portfolio, freq='d'):
        """
        Start                         2019-12-31 05:00:00+00:00
        End                           2023-12-29 05:00:00+00:00
        Period                               1007 days 00:00:00
        First Trade Start             2020-01-06 05:00:00+00:00
        Last Trade End                2023-12-29 05:00:00+00:00
        Coverage                              561 days 00:00:00
        Overlap Coverage                        0 days 00:00:00
        Total Records                                        97
        Total Long Trades                                    97
        Total Short Trades                                    0
        Total Closed Trades                                  96
        Total Open Trades                                     1
        Open Trade PnL                               857.815458
        Win Rate [%]                                       37.5
        Max Win Streak                                        5
        Max Loss Streak                                       9
        Best Trade [%]                                17.406429
        Worst Trade [%]                               -13.05181
        Avg Winning Trade [%]                          4.976603
        Avg Losing Trade [%]                          -3.157603
        Avg Winning Trade Duration              9 days 12:40:00
        Avg Losing Trade Duration               2 days 22:48:00
        Profit Factor                                  0.892886
        Expectancy                                   -20.684695
        SQN                                           -0.422253
        """
        portfolio.trades.stats()
        
    def backtest_metrics(self, portfolio, freq='d'):
        # backtest_metrics = portfolio.returns_stats() + portfolio.stats()
        stats_df = self.portfolio_stats(portfolio, freq=freq)
        returns_stats = self.portfolio_returns_stats(portfolio, freq=freq)
        backtest_df = pd.concat([stats_df, returns_stats], axis=1)
        
        backtest_df['profit_loss_ratio'] = self.profit_loss_ratio(portfolio)
        backtest_df['freq_stats'] = freq
        backtest_df['score'] = self.score(backtest_df)
        backtest_df = backtest_df.apply(lambda x: x.round(3) if x.dtype == 'float64' else x) # 保留两位小数
        return backtest_df
    
    def trade_details(self, portfolio):
        trade_details_raw_df = portfolio.trades.records_readable
        trade_details_df = trade_details_raw_df.rename(columns={'Exit Trade Id': 'exit_trade_id',
                                                             'Column': 'column',
                                                             'Size': 'size',
                                                             'Entry Timestamp': 'entry_timestamp',
                                                             'Avg Entry Price': 'avg_entry_price',
                                                             'Entry Fees': 'entry_fees',
                                                             'Exit Timestamp': 'exit_timestamp',
                                                             'Avg Exit Price': 'avg_exit_price',
                                                             'Exit Fees': 'exit_fees',
                                                             'PnL': 'pnl',
                                                             'Return': 'return',
                                                             'Direction': 'direction',
                                                             'Status': 'status',
                                                             'Position Id': 'position_id'})
        utils_data.output_database(trade_details_df, 'ads_info_incr_valid_trade_details')
        
    def dataset_demo(self, symbols,
                           date_start='2021-01-01',
                           date_end='2022-01-01',
                           column_price='Close'): # ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        
        data = vbt.YFData.download(symbols , start=date_start, end=date_end, missing_index='drop').get(column_price)
        if isinstance(data, pd.Series):  # 只有一个symbols的时候是获取不到symbol_name的
            if len(symbols)==1:
                data = data.to_frame(name=symbols[0])
            else:
                data = data.to_frame(name=symbols)
            data.columns.name = 'symbol'
        
        data = self.dataset_clean(data)
        return data
    
    def dwd_freq_full_portfolio_daily_backtest(self):
        self.data_dwd.dwd_freq_full_portfolio_daily_backtest()
    
    def dataset_clean(self, df):
        # vbt.Portfolio.from_signals()会对数据自动补全，因此全空数据error:TypingError: non-precise type array(pyobject, 2d, F)
        df = df.loc[:,~(df.count()==0)]
        
        # When calculating annualized data, the growth of the whole year is calculated based on the growth of a limited number of days. However, in the field of stocks, the number of trading days per year is fixed, so this calculation will lead to data such as annualized returns being significantly higher or lower.
        # https://github.com/polakowo/vectorbt/issues/252
        # debug: year_freq='243 days',
        # full_date_range = pd.date_range(start=df.index.min(), end=df.index.max())
        # df = df.reindex(full_date_range).fillna(method='ffill')
        return df
    
    def dataset(self, symbols, date_start='2020-01-01', date_end='2022-01-01', column_price='Close'):
        if not utils_data.table_in_database('dwd_ohlc_full_portfolio_daily_backtest_eff'):
            self.dwd_freq_full_portfolio_daily_backtest()
        with utils_database.engine_conn("POSTGRES") as conn:
            portfolio_daily_df = pd.read_sql(f"SELECT * FROM dwd_ohlc_full_portfolio_daily_backtest_eff WHERE date BETWEEN '{date_start}' AND '{date_end}'", con=conn.engine)
            #portfolio_daily_df = pd.read_sql('dwd_freq_full_portfolio_daily_backtest', con=conn.engine)
        portfolio_daily_df.columns.name = 'symbol'
        
        #portfolio_daily_df.columns.name = 'symbol'
        portfolio_daily_df.index = portfolio_daily_df.date
        portfolio_daily_df.drop(['date', 'insert_timestamp'], axis=1, inplace=True)
        if len(symbols)!=0:
            portfolio_daily_df = portfolio_daily_df[symbols]
        
        portfolio_daily_df = self.dataset_clean(portfolio_daily_df)
        return portfolio_daily_df
    
    def output_backtest_metrics(self, portfolio, data, comparison_experiment='base'):
        # Trade Details
        if self.output_trade_details:
            self.trade_details(portfolio)
    
        # Get backtest results
        backtest_df = self.backtest_metrics(portfolio)
        if isinstance(backtest_df.index, pd.MultiIndex):
            backtest_df['symbol'] = backtest_df.index.get_level_values('symbol')
            #backtest_df['fast_ma'] = backtest_df.index.get_level_values(0)
            #backtest_df['slow_ma'] = backtest_df.index.get_level_values(1)
            #backtest_df['signal_ma'] = backtest_df.index.get_level_values(2)
            #backtest_df['rsi_window'] = backtest_df.index.get_level_values('rsi_window')
        else:
            backtest_df['symbol'] = backtest_df.index
            backtest_df[['fast_ma', 'slow_ma', 'signal_ma']] = None 
        backtest_df['comparison_experiment'] = comparison_experiment
        backtest_df['fees'] = self.portfolio_params['fees']
        backtest_df['slippage'] = self.portfolio_params['slippage']
        backtest_df['freq_portfolio'] = self.portfolio_params['freq']
        backtest_df[['start', 'end']] = backtest_df[['start', 'end']].apply(lambda col: col.dt.strftime("%F %T"))
        backtest_df[['period']] = backtest_df[['period']].apply(lambda x: x.dt.days.astype(int))
        backtest_df[['max_dd_duration', 'avg_winning_trade_duration','avg_losing_trade_duration']] = backtest_df[['max_dd_duration','avg_winning_trade_duration','avg_losing_trade_duration']].apply(lambda x: x.dt.days.astype(str))# + ' days'
        backtest_df['bar_num'] = data.count().tolist()
        backtest_df['date_start'] = data.apply(lambda col: col.first_valid_index()).tolist()
        backtest_df['date_end'] = data.apply(lambda col: col.last_valid_index()).tolist()
        backtest_df['price_start'] = data.apply(lambda col: col.loc[col.first_valid_index()]).tolist()
        backtest_df['price_end'] = data.apply(lambda col: col.loc[col.last_valid_index()]).tolist()

        
        backtest_df['primary_key'] = (backtest_df['start'].astype(str) +
                                      backtest_df['end'].astype(str) +
                                      backtest_df['symbol'].astype(str) +
                                      backtest_df['freq_portfolio'].astype(str)+
                                      backtest_df['freq_stats'].astype(str)+
                                      backtest_df['fees'].astype(str)+
                                      backtest_df['slippage'].astype(str)+
                                      # backtest_df['init_cash'].astype(str)+
                                      #backtest_df['fast_ma'].astype(str)+
                                      #backtest_df['slow_ma'].astype(str)+
                                      backtest_df['comparison_experiment'].astype(str)
                                      ).apply(utils_character.md5_str) # md5
        
        backtest_df = backtest_df.rename(columns={'symbol': 'full_code',
                                                  })

        if self.output=='database':
            utils_data.output_database(backtest_df,
                                       filename=self.output_database_filename,
                                       #dtype={'fast_ma': int, #int
                                       #       'slow_ma': int, #int
                                       #       }
                                       )
        elif self.output=='csv':
            utils_data.output_local_file(backtest_df, filename='backtest_df',index=False)
        elif self.output==False:
            ...
        else:
            ...
            #return backtest_df
            
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
    
    def strategy_params(self, strategy_params_list):
        strategy_params_df = pd.DataFrame(strategy_params_list)
        strategy_params_df['primary_key'] = strategy_params_df['window_fast'].astype(str)+'-'+\
                                            strategy_params_df['window_slow'].astype(str)
        return strategy_params_df

    def backtest_chunk(self, args):
        data, strategy_params_df_chunk, comparison_experiment = args
        entries_exits_t = strategy_params_df_chunk.groupby('primary_key', as_index=False).apply(self.strategy, data)#, include_groups=False
        entries_exits = entries_exits_t.T
        entries_exits.columns = entries_exits.columns.droplevel(0)
        entries = entries_exits['entries']
        exits = entries_exits['exits']
        
        data_concat = pd.concat([data] * strategy_params_df_chunk.shape[0], axis=1)
        logger.info(f"Data shape: {data_concat.shape}")
        portfolio = vbt.Portfolio.from_signals(data_concat,
                                               entries.astype(np.bool_),
                                               exits.astype(np.bool_),
                                               **self.portfolio_params)
        self.output_backtest_metrics(portfolio, data_concat, comparison_experiment)
        return portfolio

    def backtest(self, data, comparison_experiment='base'):
        if comparison_experiment == 'base':
            portfolio = vbt.Portfolio.from_holding(close=data, **self.portfolio_params)
            logger.info(portfolio)
            self.output_backtest_metrics(portfolio, data, comparison_experiment)
        else:
            chunks = [self.strategy_params_df.iloc[i:i+self.strategy_params_batch_size] 
                      for i in range(0, len(self.strategy_params_df), self.strategy_params_batch_size)]
            
            args_list = [(data, chunk, comparison_experiment) for chunk in chunks]
            
            if self.use_multiprocess:
                # 使用多进程
                multiprocessing.set_start_method('spawn', force=True)
                with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                    portfolios = pool.map(self.backtest_chunk, args_list)
            else:
                # 使用单进程
                portfolios = [self.backtest_chunk(args) for args in args_list]
            
            # 如果需要，这里可以合并或进一步处理多个portfolio的结果
            return portfolios
        
    def ablation_experiment(self, symbols,
                            date_start='2021-01-01',
                            date_end='2022-01-01',
                            comparison_experiment=None,
                            if_exists='fail',  # ['fail','replace','append']
                            ):
        
        # 1:1 = strategy : base in dataset_demo
        portfolio_df = self.dataset_demo(symbols, date_start=date_start, date_end=date_end)
        
        base_portfolio = self.backtest(data=portfolio_df)
        strategy_portfolio = self.backtest(data=portfolio_df, comparison_experiment=comparison_experiment)
        return base_portfolio, strategy_portfolio
    
    def plot(self, fig):
        fig.write_html("./html/portfolio_plot.html")
        

if __name__ == '__main__':
    # symbols=["ADA-USD"]
    symbols=["ADA-USD", "ETH-USD"]  # "BTC-USD", 'AAPL', 'MSFT', 'GOOG'
    backtest_vectorbt = backtestVectorbt(output_trade_details=False,
                                         output='database',
                                         portfolio_params={'freq': 'd',
                                                          'fees': 0.001,  # 0.1% per trade
                                                          'slippage': 0.001,  # 0.1% slippage
                                                          'init_cash': 10000})
    
    base_portfolio, strategy_portfolio = backtest_vectorbt.ablation_experiment(symbols=symbols,
                                                                   date_start='2021-01-01',
                                                                   date_end='2022-01-01',
                                                                   comparison_experiment='10-50 Dual Moving Average',  # 10/50双线
                                                                   )


# data = backtest_vectorbt.dataset(symbols=["ADA-USD", "ETH-USD"], date_start=date_start, date_end=date_end)

#https://github.com/polakowo/vectorbt/blob/54cbe7c5bff332b510d1075c5cf11d006c1b1846/vectorbt/portfolio/trades.py#L69
#胜率，盈亏比，plot_pnl

# =============================================================================
# Daily and longer timeframes: freq
# 
# 'D' : Daily
# 'B' : Business day
# 'W' : Weekly
# 'ME' : Month end
# 'MS' : Month start
# 'Q' : Quarter end
# 'QS' : Quarter start
# 'A' or 'Y' : Year end
# 'AS' or 'YS' : Year start
# Hourly and shorter timeframes:
# 
# 'H' : Hourly
# 'T' or 'min' : Minutes
# 'S' : Seconds
# 'L' or 'ms' : Milliseconds
# 'U' or 'us' : Microseconds
# 'N' : Nanoseconds
# Offset Modifiers:
# 
# '5T' : 5 minutes
# '30T' : 30 minutes
# '15min' : 15 minutes
# '2H' : 2 hours
# =============================================================================
