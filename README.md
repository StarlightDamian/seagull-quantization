# 海鸥量化

**成员**：俞老师

**频率**：日频、15分钟频

**语言**：Python

<p align="left">
    <img alt="ViewCount" valign="bottom" src="https://views.whatilearened.today/views/github/UFund-Me/Qbot.svg">
    <a href='https://github.com/MShawon/github-clone-count-badge'><img alt='GitHub Clones' valign="bottom" src='https://img.shields.io/badge/dynamic/json?color=success&label=Clone&query=count&url=https://gist.githubusercontent.com/MShawon/cf89f3274d06170b8a4973039aa6220a/raw/clone.json&logo=github'></a>
    <img alt="releases" valign="bottom" src="https://img.shields.io/github/downloads/UFund-Me/Qbot/total"> <code>since Sep 26</code>
</p>

[![CodeQL](https://github.com/UFund-Me/Qbot/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/UFund-Me/Qbot/actions/workflows/codeql-analysis.yml)
[![AutoTrade](https://github.com/UFund-Me/Qbot/actions/workflows/auto-trade.yml/badge.svg)](https://github.com/UFund-Me/Qbot/actions/workflows/auto-trade.yml)
[![Pylint](https://github.com/UFund-Me/Qbot/actions/workflows/pylint.yml/badge.svg)](https://github.com/UFund-Me/Qbot/actions/workflows/pylint.yml)
[![Coverage](https://github.com/UFund-Me/Qbot/actions/workflows/coverage.yml/badge.svg)](https://github.com/UFund-Me/Qbot/actions/workflows/coverage.yml)
<a href="https://github.com/UFund-Me/Qbot"><img src="https://img.shields.io/badge/Python-%203.8|%203.9-000000.svg?logo=Python&color=blue" alt="Python version"></a>
<a href="https://ufund-me.github.io/Qbot/#/"><img src="https://readthedocs.org/projects/pyod/badge/?version=latest" alt="Documentation status"></a>

```
![GitHub stars](https://img.shields.io/github/stars/username/repo?style=social)
```

1. 

```markdown
![Static Badge](https://img.shields.io/badge/:badgeContent)
```

Execute

（https://img.shields.io/badge/any_text-you_like_damian-blue）

# 内测

**内测周期**：2023/12/27  ——  2024/1/5

**内测目的**：

* 补全用户最需要的功能

* 测试数据BUG

* 

# 版本更新记录

|   项目    |  版本  |  上线日期  |                           新增功能                           |                备注                |
| :-------: | :----: | :--------: | :----------------------------------------------------------: | :--------------------------------: |
| fish-leap | v0.1.0 | 2023-12-27 | 1.单一股票进行股价预测<br />2.单一股票的买入点和卖出点<br />3.训练一只股票模型速度为15分钟 |                                    |
| fish-leap | v0.2.0 | 2024-02-22 | 1.不再进行单一股票的训练与预测，而且把整个股市全天的数据为单位进行训练与预测。<br/>2.重点优化推票的逻辑，进行每日TOP5的推票，进行多次回测，优化推票逻辑。 |                                    |
| fish-leap | v0.2.1 | 2024-02-26 | 1.接入15分钟线数据<br />2.首页按推荐排序，推荐效果优化至可用<br />3.去除成交额为0和一字涨跌停数据预测，减少干扰项 |                                    |
| fish-leap | v0.2.2 | 2024-03-26 | 1.分类为主板、创业板、科创版、新三板、北交所，在模型训练和预测阶段限制每日涨跌幅为相应比例 | 修改了训练、评估、测试阶段的逻辑。 |



# 开放

**开放时间**：2024/1/5  ——  2024/12/31

**投放平台流量入口**：小红书 / 抖音 / 知乎

**私域流量**：微信群



# 私域流量

| 手机号 | 平台 |        名称        | 价格 | 新增功能 | 备注 |
| :----: | :--: | :----------------: | :--: | :------: | :--: |
|        | 微信 | 俞老师量化(内测版) |      |          |      |
|        | 微信 |   俞老师量化2024   |      |          |      |
|        |      |                    |      |          |      |



# 功能

**功能**：

	1.输入**股票代码**，返回对应的推荐买入/卖出价格
	
	2. 每日晚上6点，提供全部A股下一个交易日的预估最高点、预估最低点。（日频）

**待开放功能**：

1. 在适合买入/卖出的时候，提供**指定股票代码**的推荐买入价格/卖出价格
2. 提供该策略的历史回测图

**待开发功能**：

1. 在适合买入/卖出的时候，提供**全部股票代码**的推荐买入价格/卖出价格

2. 提供**全部股票代码**该策略的历史回测图

3. 补充**宏观经济指标**作为特征

4. **模糊搜索**

5. 日线输出推荐，修改为**15分钟线**推荐

6. **微信自动回复**

7. 做这个拉群最终的目的是 你要根据板块的分析 或者当天的新闻筛选出最优化的几只

8. 推票就是这样 是提供买哪只 而不是只单单的预测我自己要买的价格

   

# 优化方向

1.时间维度

2.推票纬度：我的推票逻辑是：
不是那种一字板的，那种一般人也参与不了，而是预测价格差值最大的，普通人可以在低价买入，高价卖出的那种

3.仓位

4.投资组合

5.止盈止损

6.最佳买卖时机

7.风险预测

8.确定性程度

9新股，用超过半年的数据进行训练

10持仓比例和成本

11.季报、年报



1.通过后向前来评估，当前价格是否处于短期/中期/长期的价格硅地，低于95%的时候

2.评估指标小于x，x=2y

3.双曲线日线交叉少的

持仓分布

主力流入流出

板块分析，板块内流动性最好，市盈率最低，

大基金持仓

宏观数据

美元降息、美元指数

流动性

用未来五天的数据进行验证，五日最低价

大部分人亏钱的方向相反走

对应的期货会先于股票有表现，如航运

如果五分钟线的流动性有异常（k<0），果断清仓，等稳定上行再入手

如工业富联和美股英伟达强关联

先判断当前行情的类型，再根据行情是趋势/震荡配置对应的策略

不同的资金规模会遇到不同的问题。小资金是手续费、大资金是流动性

旋转180度看K线

通过过去五天的线预测未来五天，而不是只用今日

用rank来做

用图来做

用模型做投资组合rank

涨的概率 * 涨的幅度>跌的概率 * 跌的幅度 * 2

涨跌的稳不稳，还是突然暴涨

涨的量能是否能跟上涨幅

赛马轮动机制

追新闻热点

看看涨跌翻转，判别模型是否能区分

比如小市值策略以前是α，现在是风险策略，训练时候要考虑权重衰减，或者冻结最后一层，用最近20天的数据去微调模型

rank后是否能够以单笔直接预测收益率和胜率进行排序

# 预警模块

1.ST、退市

2.季报

3.收缩

# 问题记录

1.没有ETF板块数据

2.有部分涨跌幅溢出（超出涨跌停）

3.没有历史数据的次新股怎么进行两年前的回测

4.最近上市的数据剔除，没有经过训练

5.短线和长线的百分比收益是有区别的。

6.上市后多久的数据才算正常数据能够用于训练和预测

7.rear_close_pct_pred差距很大



# 问题

1.为什么交易是对称的，盈亏同源

# 特征

1.炸板（封板没封死）、封板

2.MACD

3.主力

4.30日最低、90日最低、1年最低、2年最低、1年最高

# 返回

1. 推票
2. 回测
3. 买入/卖出价格




# 评估指标

1.收益/风险

2.胜率/时间/成本

# 权责

**责任**：

* 只提供数据分析
* 不指导交易
* 不承担责任



# 长期主义

1.盈亏比

2.行业板块 & 龙头

3.趋势开始
