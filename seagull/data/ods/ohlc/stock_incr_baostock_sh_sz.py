# -*- coding: utf-8 -*-
"""
@Date: 2023/8/8 13:07
@Author: Damian
@Email: zengyuwei1995@163.com
@File: stock_incr_baostock_sh_sz.py
@Description: 获取指定日期全部股票的日K线数据(ods/ohlc/stock_incr_baostock_sh_sz)
@Update cycle: day
code_name 不属于特征，在这一层加入
5、15、30、60分钟线指标参数(不包含指数)
"""
import os
import asyncio

import baostock as bs
import pandas as pd

from seagull.settings import PATH
from seagull.utils import utils_database, utils_log


log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')


class OdsOhlcStockIncrBaostockShSz:
    """
    A股的K线数据，全量历史数据接口 (异步封装)
    """

    def __init__(self):
        with utils_database.engine_conn("POSTGRES") as conn:
            self.ods_stock_base_df = pd.read_sql(
                "select code from ods_info_stock_incr_baostock", con=conn.engine
            )
        # code = pd.read_sql("select distinct code from ods_ohlc_incr_baostock_stock_sh_sz_daily", con=conn.engine)
        # self.ods_stock_base_df = self.ods_stock_base_df[~(self.ods_stock_base_df.code.isin(code))]

    # @staticmethod
    def _sync_query(self, code, date_start, date_end,
                    fields='date,time,code,open,high,low,close,volume,amount',
                    frequency='5',
                    adjustflag='2') -> pd.DataFrame:
        """同步行为：调用 baostock，返回 DataFrame"""
        logger.info(
            f"[SYNC] start code={code}, {date_start}~{date_end}, "
            f"fields={fields}, "
            f"freq={frequency}, adj={adjustflag}"
        )
        rs = bs.query_history_k_data_plus(
            code,
            fields=fields,
            start_date=date_start,
            end_date=date_end,
            frequency=frequency,
            adjustflag=adjustflag
        )
        try:
            df = rs.get_data()
        except Exception:
            logger.exception(f"baostock error for {code}")
            return pd.DataFrame()
        if df.empty:
            logger.warning(f"[SYNC] {code} returned empty")
        else:
            logger.info(f"[SYNC] {code} got {df.shape} rows")
        return df

    async def stock_sh_sz_1(
            self,
            code: str,
            date_start: str,
            date_end: str,
            fields: str ='date,time,code,open,high,low,close,volume,amount', #'date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,psTTM,pcfNcfTTM,pbMRQ,isST',
            frequency: str = '5',
            adjustflag: str = '3'
    ) -> pd.DataFrame:
        """
        异步接口：并发调用 _sync_query
        """
        # 把同步查询放到默认线程池里
        df = await asyncio.to_thread(
            self._sync_query,
            code, date_start, date_end, fields, frequency, adjustflag
        )
        return df

    # def stock_sh_sz(self, date_start='1990-01-01',
    #                 date_end='2100-01-01',
    #                 fields='date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,psTTM,pcfNcfTTM,pbMRQ,isST',
    #                 frequency='d',
    #                 adjustflag='2'):
    #     bs.login()
    #
    #     df = self.ods_stock_base_df.groupby('code').apply(self.stock_sh_sz_1,
    #                                                       date_start=date_start,
    #                                                       date_end=date_end,
    #                                                       fields=fields,
    #                                                       frequency=frequency,
    #                                                       adjustflag=adjustflag)
    #     df['freq'] = frequency
    #
    #     # adjustment_type as adj_type in ['None', 'pre', 'post'], 复权状态(1：后复权， 2：前复权，3：不复权）
    #     adj_type_dict = {"1": "post",
    #                      "2": "pre",
    #                      "3": None}
    #     df['adj_type'] = adj_type_dict.get(adjustflag)
    #
    #     bs.logout()
    #     return df.reset_index(drop=True)


def base_test():
    lg = bs.login()
    # 显示登陆返回信息
    print('login respond error_code:'+lg.error_code)
    print('login respond  error_msg:'+lg.error_msg)
    # 详细指标参数，参见“历史行情指标参数”章节；“分钟线”参数与“日线”参数不同。“分钟线”不包含指数。
    # 分钟线指标：date,time,code,open,high,low,close,volume,amount,adjustflag
    # 周月线指标：date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg
    rs = bs.query_history_k_data_plus(
        "sh.600000",  # "sz.000001",
        # "date,time,code,high,low,close,volume",
        # "date,time,code,open,high,low,close,volume,amount,adjustflag",
        # fields='date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,psTTM,pcfNcfTTM,pbMRQ,isST',
        fields='date,time,code,high,low,close,volume',
        start_date='2023-01-01', end_date='2023-06-30',
        frequency="60", adjustflag="3")
    print('query_history_k_data_plus respond error_code:'+rs.error_code)
    print('query_history_k_data_plus respond  error_msg:'+rs.error_msg)

    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)

    print(result)
    bs.logout()


def unit_test():
    ods = OdsOhlcStockIncrBaostockShSz()
    with utils_database.engine_conn("POSTGRES") as conn:
        ods.ods_stock_base_df = pd.read_sql("select code from ods_info_incr_baostock_stock_base", con=conn.engine).sample(n=10)
    stock_sh_sz_daily_df = ods.stock_sh_sz()
    print(stock_sh_sz_daily_df)

    # from seagull.utils import utils_data
    # utils_data.output_database(stock_sh_sz_daily_df,
    #                            filename='ods_ohlc_incr_baostock_stock_sh_sz_daily')


async def main():
    api = OdsOhlcStockIncrBaostockShSz()
    codes = ['sh.300588', 'sh.000001', 'sz.000001']  # 示例
    tasks = [
        api.stock_sh_sz_1(code=code,
                          date_start='2025-01-01',
                          date_end='2025-06-30',
                          fields="date,time,code,high,low,close,volume")
        for code in codes
    ]
    results = await asyncio.gather(*tasks)
    for code, df in zip(codes, results):
        print(code, df.shape)


if __name__ == '__main__':
    # base_test()

    lg = bs.login()
    asyncio.run(main())
    bs.logout()

