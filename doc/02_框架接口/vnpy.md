下面给出几个在 vn.py 中常见的使用场景示例，全部用 pandas 来“假装”喂入数据和接收结果，方便你在无实盘／无行情接口时进行逻辑调试。

------

## 1. 用 DataFrame 作为历史 K 线，跑回测

```python
import pandas as pd
from vnpy.app.cta_backtester import BacktestingEngine, OptimizationSetting

# 模拟日线数据：date, open, high, low, close, volume
dates = pd.date_range("2023-01-01", periods=100)
bars = pd.DataFrame({
    "date": dates,
    "open": 100 + pd.np.sin(pd.np.arange(100)/10),
    "high": 100 + pd.np.sin(pd.np.arange(100)/10) + 0.5,
    "low":  100 + pd.np.sin(pd.np.arange(100)/10) - 0.5,
    "close":100 + pd.np.sin(pd.np.arange(100)/10),
    "volume": pd.np.random.randint(100, 200, 100),
})
bars.set_index("date", inplace=True)

# 1) 初始化回测引擎
engine = BacktestingEngine()
engine.set_parameters(
    vt_symbol="TEST.STK",
    interval="1d",
    start=bars.index[0],
    end=bars.index[-1],
    rate=0.0003,
    slippage=0.01,
    size=1,
    pricetick=0.01,
    capital=100000
)

# 2) 加载 DataFrame 到引擎
engine.load_data(bars)

# 3) 简易策略：突破开盘价后做多
from vnpy.app.cta_strategy import CtaTemplate, StopOrder, TickData, BarData

class TestStrategy(CtaTemplate):
    author = "Demo"
    parameters = ["fixed_size"]
    variables = []
    fixed_size = 1

    def __init__(self, engine, strategy_name, vt_symbol, setting):
        super().__init__(engine, strategy_name, vt_symbol, setting)
        self.open_price = None

    def on_init(self):
        self.write_log("策略初始化")

    def on_start(self):
        self.write_log("策略启动")

    def on_bar(self, bar: BarData):
        if self.open_price is None:
            self.open_price = bar.open_price
        if bar.close_price > self.open_price * 1.01 and not self.pos:
            self.buy(bar.close_price, self.fixed_size)

# 4) 添加策略并运行
engine.add_strategy(TestStrategy, {"fixed_size": 1})
engine.run_backtesting()
df_result = engine.calculate_result()
df_statistics = engine.calculate_statistics()

print(df_result.head())
print(df_statistics)
```

**输出示例**：

```text
             trade_time  direction  price  volume   pnl
0 2023-01-10 00:00:00       LONG  101.23       1  1.23
1 2023-01-20 00:00:00       LONG  102.34       1  2.34
...
```

------

## 2. 用 Tick Data 模拟撮合与下单

```python
import pandas as pd
from vnpy.trader.object import TickData, OrderData, TradeData
from vnpy.app.cta_strategy import CtaTemplate, CtaEngine

# 生成模拟 Tick 序列
times = pd.date_range("2023-07-01 09:30", periods=10, freq="S")
ticks = []
for t in times:
    ticks.append(TickData(
        symbol="TEST.STK",
        exchange="SSE",
        datetime=t.to_pydatetime(),
        name="Test",
        last_price=100 + 0.01*(t.second),
        volume=1000 + t.second,
        bid_price_1=99.9,
        ask_price_1=100.1
    ))

# 简易 CTA 引擎 + 策略
class TickStrategy(CtaTemplate):
    def __init__(self, engine, strategy_name, vt_symbol, setting):
        super().__init__(engine, strategy_name, vt_symbol, setting)

    def on_tick(self, tick: TickData):
        if tick.last_price > 100.05 and self.pos == 0:
            self.buy(tick.last_price, 1)

# 构造 Engine
engine = CtaEngine(None, None, None)
strategy = engine.add_strategy(TickStrategy, {}, "TEST.STK")
engine.init_engine()
engine.start_all()

# 推送 ticks
for tk in ticks:
    engine.process_tick(tk)

# 查看下单记录
for order in engine.get_all_active_orders():
    print(order)

for trade in engine.get_all_trades():
    print(trade)
```

------

## 3. 优化示例：Grid Search

```python
from vnpy.app.cta_backtester import BacktestingEngine, OptimizationSetting

# 假设已有 bars DataFrame（同示例1）
engine = BacktestingEngine()
engine.set_parameters(...)

# 设置策略和待调参数
opt_setting = OptimizationSetting()
opt_setting.set_target("sharpe_ratio")
opt_setting.add_parameter("TestStrategy", "fixed_size", 1, 5, 1)

# 运行优化
engine.add_strategy(TestStrategy, {})
engine.load_data(bars)
df_opt = engine.run_optimization(opt_setting)

print(df_opt.head())
```

**输出示例**（每行一个参数组合的回测指标）：

| fixed_size | total_return | sharpe_ratio | max_drawdown |
| ---------- | ------------ | ------------ | ------------ |
| 1          | 0.12         | 1.05         | 0.15         |
| 2          | 0.10         | 0.98         | 0.12         |
| …          | …            | …            | …            |

------

以上三个示例覆盖了 vn.py 中最常见的“历史回测”、“Tick 下单”、“参数优化”场景，并全部用 pandas 来构造或展示结果，便于在没有真实行情和交易接口的环境下快速验证策略逻辑。