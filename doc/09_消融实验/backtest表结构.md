| 字段中文名 | vectorbt - portfolio.stats() | 数据库字段 | 基准_示例数据 | 策略_示例数据 | 备注 |
| :----------: | :--------: | ------------ | ------------ | ------------ | ------------ |
|开始时间        |Start        | date_start | 2020-12-31 05:00:00+00:00 |                 2020-12-31 05:00:00+00:00|                 |
|结束时间           |End           | date_end | 2021-12-31 05:00:00+00:00 |                2021-12-31 05:00:00+00:00|                |
|时间长度         |Period         | period | 253 days 00:00:00 |                       253 days 00:00:00|                       |
|初始金额    |Start Value    | start_value | 10000.0 |                                 10000.0|                                 |
|结束金额          |End Value          | end_value | 13464.81 |                        11449.736673|                        |
|总回报 [%]                  |Total Return [%]                  | total_return | 34.64 |            14.497367|            |
|基准回报 [%]               |Benchmark Return [%]               | benchmark_return | 34.64 |           50.765239|           |
|最大总风险敞口 [%]              |Max Gross Exposure [%]              | max_gross_exposure | 100.0 |              100.0|              |
|总费用已支付                 |Total Fees Paid                 | total_fees_paid | 0.0 |                    0.0|                    |
|最大回撤 [%]               |Max Drawdown [%]               | max_dd | 18.59 |                7.865672|                |
|最大回撤持续时间         |Max Drawdown Duration         | max_dd_duration | Timedelta('111 天 00:00:00') |         43 days 08:00:00| 回到当前最高值用了多少天 |
|总交易数                        |Total Trades                        | total_trades | 1 |                3.0|                |
|总平仓交易数         |Total Closed Trades         | total_closed_trades | 0 |                   2.333333|                   |
|总未平仓交易数               |Total Open Trades               | total_open_trades | 1 |               0.666667|               |
|未平仓交易 PnL              |Open Trade PnL              | open_trade_pnl | 3464.81 |                1017.590773|                |
|胜率 [%]                    |Win Rate [%]                    | win_rate | nan |              44.444444|              |
|最佳交易 [%]                 |Best Trade [%]                 | best_trade | nan |                8.427613|                |
|最差交易 [%]                 |Worst Trade [%]                 | worst_trade | nan |             -3.741962|             |
|平均获胜交易 [%]          |Avg Winning Trade [%]          | avg_winning_trade | nan |                8.427613|                |
|平均亏损交易 [%]               |Avg Losing Trade [%]               | avg_losing_trade | nan |           -3.355587|           |
|平均获胜交易持续时间      |Avg Winning Trade Duration      | avg_winning_trade_duration | NaT |       61 days 00:00:00|       |
|平均亏损交易持续时间       |Avg Losing Trade Duration       | avg_losing_trade_duration | NaT |       13 days 20:00:00|       |
|利润因子                   |Profit Factor                   | profit_factor | nan |               2.390847|               |
|预期                    |Expectancy                    | expectancy | nan |               178.292628|               |
|夏普比                |Sharpe Ratio                | sharpe_ratio | 1.57 |                    1.09306|                    |
|卡尔玛比率                 |Calmar Ratio                 | calmar_ratio | 2.88 |                  2.540644|                  |
|欧米茄比率                |Omega Ratio                | omega_ratio | 1.24 |                    1.244874|                    |
|sortino风险比                 |Sortino Ratio                 | sortino_ratio | 2.35 |                 1.658201| 越大风险越小 |
|盈亏比 | | profit_loss_ratio |  | | |
|年化收益 | | annual_return | | | |