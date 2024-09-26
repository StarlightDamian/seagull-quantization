## lib说明

|            类别            |           python文件            |              说明               |                            备注                             |
| :------------------------: | :-----------------------------: | :-----------------------------: | :---------------------------------------------------------: |
|  **ablation_experiment**   |                                 |            消融实验             |                                                             |
|      **application**       |                                 |           应用文件夹            |  在这一层通过控制变量来对指定金额、策略、历史数据进行回测   |
|      **backtesting**       |                                 |         量化回测文件夹          |                                                             |
|          **base**          |                                 |           基础文件夹            |                      工具包和数据获取                       |
|            base            |       **base_arguments**        |            参数管理             |                                                             |
|            base            |    **base_connect_database**    |           链接数据库            |                                                             |
|            base            |         **base_utils**          |             工具包              |                                                             |
|  **feature_engineering**   |                                 |         特征工程文件夹          |                                                             |
|        **get_data**        |                                 |         获取数据文件夹          |                                                             |
|          get_data          |          **data_get**           |            获取数据             |                                                             |
|          get_data          |      **data_distribution**      |            数据分发             |                                                             |
|          get_data          | **data_history_a_stock_k_data** | 获取指定日期全部股票的日K线数据 |                                                             |
|          get_data          |        **data_loading**         |            数据加载             |                                                             |
|          get_data          |         **data_plate**          |       获取股票对应的板块        |                                                             |
|         **other**          |                                 |            其他代码             |                                                             |
| **reinforcement_learning** |                                 |         强化学习文件夹          |                                                             |
|   reinforcement_learning   |   reinforcement_learning_ddpg   |            DDPG算法             |                                                             |
|   reinforcement_learning   |   reinforcement_learning_sac    |             SAC算法             |                                                             |
|        **strategy**        |                                 |         量化策略文件夹          |                                                             |
|         **trade**          |                                 |           交易文件夹            |                                                             |
|           trade            |         **trade_eval**          |       交易评估指标的计算        |                                                             |
|           trade            |     **trade_handling_fee**      |           手续费计算            |                                                             |
|           trade            |  **trade_investment_strategy**  |            投资策略             |                                                             |
|           trade            |       **trade_portfolio**       |            投资组合             |            通过投资组合来获得更稳定的收益和复利             |
|           trade            |         **trade_relu**          |            交易规则             | ETF交易没限制，普通股票>100股<br />周末不交易、节假日不交易 |
|           trade            |    **trade_risk_management**    |            风险管理             |                  评估股票、交易的风险程度                   |
|         **train**          |                                 |           训练文件夹            |                                                             |
|           train            |         **train_main**          |           训练主程序            |                                                             |
|     **visualization**      |                                 |          可视化文件夹           |                                                             |
|                            |                                 |                                 |                                                             |



