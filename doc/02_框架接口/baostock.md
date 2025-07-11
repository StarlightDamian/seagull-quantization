你之所以把“异步”都套了一圈，速度还是和原始串行差不多，核心原因就在于：

1. **Baostock SDK 本身是同步阻塞**，并且你在调用它的地方又额外加了一个全局 `Lock`，这意味着同一时刻只有一个请求在跑。
2. `asyncio.to_thread` 也只是把同步调用丢到线程池，**但因为有锁**，别的线程拿不到锁，就只能排队等。
3. 所以不管你怎么改 `Semaphore`、怎么用 `asyncio.as_completed`，拉接口这一段永远只能一个一个地跑——整个流程仍然是串行的。

------

## 进一步提速的思路

### 1. **多进程并发**：绕过全局锁

由于 Baostock SDK 在单个 Python 进程里只能串行调用，一个常见的变通是 **用多个进程**，让每个子进程各自维护一份 SDK 内部状态（各自登录、各自锁），从而实现真正的并行。举个简单的框架：

```python
import os, time, multiprocessing as mp
import pandas as pd
import baostock as bs
from seagull.utils.api.utils_api_baostock import split_baostock_code
from seagull.utils import utils_data, utils_time, utils_database, utils_log

def init_worker():
    """每个进程启动时登录一次"""
    bs.login()

def fetch_task(param):
    """子进程里同步调用，无需额外锁"""
    code, ds, de = param['code'], param['date_start'], param['date_end']
    rs = bs.query_history_k_data_plus(
        code,
        fields="date,time,code,high,low,close,volume",
        start_date=ds,
        end_date=de,
        frequency="5",
        adjustflag="2"
    )
    return rs.get_data()

def main():
    # 日志、参数准备同之前
    with utils_database.engine_conn("POSTGRES") as conn:
        codes = pd.read_sql("SELECT code FROM ods_info_stock_incr_baostock", con=conn.engine)['code'].tolist()
    batch = utils_time.make_param_grid(
        date_start="2024-01-01",
        date_end=str(pd.Timestamp.today().date()),
        window_days=60,
        code=codes
    )
    # 例如取 2000 条测试
    params = batch[:2000]

    start = time.time()
    # 启动 4 个进程并行
    with mp.Pool(processes=4, initializer=init_worker) as pool:
        dfs = pool.map(fetch_task, params)
    df_all = pd.concat(dfs, ignore_index=True)
    mid = time.time()
    print(f"Fetch done in {mid-start:.1f}s, rows={len(df_all)}")

    # 后续拆码、转换、入库同之前一次做完
    df_all = split_baostock_code(df_all)
    # ... astype/round/output_database_large ...
    end = time.time()
    print(f"Post-processing + DB write took {end-mid:.1f}s, total {(end-start):.1f}s")
```

* **优点**：每个进程各自 `bs.login()`，各自串行，但因为有 N 个进程，能达到近 N 倍的吞吐。
* **注意**：要根据机器的核心数／内存调进程数，一般 2–8 个足矣。

------

### 2. **批量 HTTP 接口（如果可行）**

有同学发现 Baostock 的 SDK 在内部也是调用一个 HTTP 服务，理论上可以直接绕过它，自己用 `aiohttp` 或 `httpx.AsyncClient` 去构造批量请求。这样你就能在一个进程、一个事件循环里同时跑上百个请求，无需锁，也不用多进程。但这需要：

* 仔细 reverse-engineer Baostock 的请求格式、鉴权方式。
* 自己实现对分页、签名、错误重试的逻辑。

如果你愿意做这部分工程量，**网络 I/O 并行度**可以直接上千，速度飞起。

------

### 3. **混合策略**

* **抓取层用多进程**（每个进程用线程池或同步调用完成自己的任务集），
* **后处理层在主进程** 批量做 `split_baostock_code`、`.astype`、`.round()`、`COPY` 写库。

这样既能突破单进程锁，又保证后处理一次性完成、内存／I/O 最优。

------

## 总结

* **当前所有“优化”之所以没效果**，是因为拉接口本身还是串行，没真正并行化。
* **最快的改造**：用 `multiprocessing.Pool` 或 `ProcessPoolExecutor`，每个子进程独立登录、独立拉取，进程数按 CPU 核心数和网络带宽来定。
* **如果有更高要求**：直接用原生 HTTP 客户端（`aiohttp`/`httpx`），自己写批量异步请求。

按上述思路一改，你就能把整体耗时从原来的“抓 1000 条要 1000×T”降到“4 ‑ 8 个进程并行，每进程抓 250 条”，加速 4–8 倍。