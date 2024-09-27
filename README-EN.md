[中文版](./README.md)

<p align="center">
    <a href="https://pypi.org/project/vectorbt" alt="PyPi">
        <img src="https://img.shields.io/pypi/v/vectorbt?color=blueviolet" />
    </a>
    <a href="https://github.com/polakowo/vectorbt/blob/master/LICENSE.md" alt="License">
	<img src="https://img.shields.io/badge/license-Fair%20Code-yellow" />
    </a>
    <a href="https://codecov.io/gh/polakowo/vectorbt" alt="codecov">
        <img src="https://codecov.io/gh/polakowo/vectorbt/branch/master/graph/badge.svg?token=YTLNAI7PS3" />
    </a>
    <a href="https://vectorbt.dev/" alt="Website">
        <img src="https://img.shields.io/website?url=https://vectorbt.dev/" />
    </a>
    <a href="https://mybinder.org/v2/gh/polakowo/vectorbt/HEAD?urlpath=lab" alt="Binder">
        <img src="https://img.shields.io/badge/launch-binder-d6604a" />
    </a>
    <a href="https://gitter.im/vectorbt/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge" alt="Join the chat at https://gitter.im/vectorbt/community">
        <img src="https://badges.gitter.im/vectorbt.svg" />
    </a>
</p>
<p align="center">
    <a href="https://pypi.org/project/vectorbt" alt="Python Versions">
        <img src="https://img.shields.io/pypi/pyversions/vectorbt.svg?logo=python&logoColor=white" />
    </a>
</p>

## Seagull Quantization

```
 seagull = Data + Strategy + Backtesting + Visualization + Automated Trading
            |         |           |              |                 |
            |         |           |              |                  \_ vnpy
            |         |           |               \____________ seaborn
            |         |            \___________________ vectorbt
            |         \___________________________ qlib, lightgbm
            \__________________________________ adata, baostock, efinance
```



## Version Updates

|          Version          |    Date    |                         New Features                          |                         Notes                          |
| :-----------------------: | :--------: | :----------------------------------------------------------: | :----------------------------------------------------: |
|  v0.1.0_20231227_alpha   | 2023-12-27 | 1. Single stock price prediction<br />2. Buy and sell points for a single stock<br />3. Training a single stock model takes 15 minutes |                                                        |
|   v0.2.0_20240222_beta   | 2024-02-22 | 1. No longer training and predicting for a single stock, but using the entire market's daily data for training and prediction.<br/>2. Focus on optimizing stock recommendation logic, providing daily TOP5 recommendations, and performing multiple backtests to optimize the recommendation logic. |                                                        |
|   v0.2.1_20240226_rc     | 2024-02-26 | 1. Integrate 15-minute interval data<br />2. Sort homepage by recommendations, optimize recommendation effect for usability<br />3. Remove data with zero trading volume and one-way limit up/down to reduce interference |                                                        |
| v0.2.2_20240222_release  | 2024-03-26 | 1. Classify as main board, ChiNext, STAR Market, NEEQ, and Beijing Stock Exchange, limiting daily price fluctuations to corresponding ratios during model training and prediction stages | Modified logic for training, evaluation, and testing.  |
|   v0.3.0_20240925_beta   | 2024-09-25 |    1. Add vectorized backtesting for fast batch historical data testing    |                                                        |

## Initialization

* Python

* ```bash
  $ git clone https://github.com/StarlightDamian/lr-camera-presets.git
```

* Configure database information

  /seagull-quantization/conf/setting_global.txt



## Quick Start

* Server submission, daily scheduled incremental data retrieval, output to database

```bash
$ cd seagull-quantization
$ nohup python main.py & > /log/main.log 2>&1 & 
```

* Input **stock code**, return recommended buy/sell price for the next trading day

```bash
$ python main.py 
```

* Backtest historical data for multiple stocks using MACD

```bash
$ python /lib/backtest/backtest_vectorbt.py 
	--strategy macd
    --date_start 2019-01-01
    --date_end 2023-01-01
    --full_code SH.510300
```



## Development Plan

| Priority |                           Feature                            | Notes |
| :------: | :----------------------------------------------------------: | :---: |
|    1     | Provide recommended buy/sell prices for **specified stock codes** when suitable for buying/selling |       |
|    1     |          Provide historical backtest chart for the strategy          |       |
|    2     | Provide recommended buy/sell prices for **all stock codes** when suitable for buying/selling |       |
|    2     |    Provide historical backtest charts for **all stock codes** using this strategy    |       |
|    2     |        Add **macroeconomic indicators** as features         |       |
|    3     |                     **Fuzzy search**                      |       |
|    3     |  Change daily output recommendations to **15-minute interval** recommendations  |       |
|    4     |                  WeChat automatic reply                   |       |
|    4     | The ultimate goal of this group is to select the most optimized stocks based on sector analysis or daily news |       |
|    4     | Stock recommendations should provide which stocks to buy, not just predict the price of stocks I want to buy |       |



## Optimization Directions

| Category |         Feature          | Notes |
| :------: | :----------------------: | :---: |
| Portfolio |        Portfolio         |       |
| Macro Features | Quarterly and Annual Reports |       |
| Features | Holding Ratio and Cost |       |
| Features | Major Capital Inflow and Outflow |       |
|  Model   |      Using Ranking       |       |
| Features |   Small Cap Strategy    |       |
| Features | Impact of Position on Strategy |       |



## Flowcharts

<img src="./image/流程图.png" alt="Flowchart" style="width:55%;"/>

<img src="./image/数据库概览.png" alt="Database Overview" style="width:55%;"/>

<img src="./image/数据获取流程.png" alt="Data Acquisition Process" style="width:55%;"/>



## References

[1] https://github.com/polakowo/vectorbt/tree/54cbe7c5bff332b510d1075c5cf11d006c1b1846

[2] https://efinance.readthedocs.io/en/latest/

```
@article{seagull-quantization,
  author = {Starlight Damian},
  title = {seagull-quantization: Local quantitative research platform},
  year = {2024}
}
```
