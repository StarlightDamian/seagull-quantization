# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 22:01:29 2024

@author: awei
业绩预报(ods_acct_incr_baostock_earnings_guidance_api)
"""
import os
import argparse

import baostock as bs
import pandas as pd
from datetime import datetime

from seagull.settings import PATH
from seagull.utils import utils_database, utils_log, utils_data

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')

class odsBaostockEarningsForecastApi:
    def __init__(self):
        with utils_database.engine_conn("POSTGRES") as conn:
            self.ods_stock_base_df = pd.read_sql("ods_info_incr_baostock_stock_base", con=conn.engine)# 获取指数、股票数据
    
    def earnings_forecast_1(self, substring,
                            date_start='2024-01-01',
                            date_end=datetime.today().strftime('%F')):
        code = substring.name
        logger.debug(f'code: {code}| date_start: {date_start}| date_end: {date_end}')
        k_rs = bs.query_forecast_report(code,
                                        start_date=date_start,
                                        end_date=date_end)
        try:
            data_df = k_rs.get_data()
        except:
            logger.error(code)
        if data_df.empty:
            logger.debug(f'{code} empty')
        else:
            logger.info(f'{code} {data_df.shape}')
            return data_df
       
    def earnings_forecast(self, date_start, date_end):
        bs.login()
        data_df = self.ods_stock_base_df.groupby('code').apply(self.earnings_forecast_1,
                                                               date_start=date_start,
                                                               date_end=date_end)
        bs.logout()
        return data_df.reset_index(drop=True) 

    def pipeline(self, date_start='1990-01-01', date_end='2100-01-01'):
        earnings_forecast_df = self.earnings_forecast(date_start=date_start,
                                                      date_end=datetime.today().strftime('%F'))
        utils_data.output_database(earnings_forecast_df, 'ods_info_incr_baostock_earnings_forecast')
        
    def output_csv(self):
        with utils_database.engine_conn("POSTGRES") as conn:
            earnings_forecast_df = pd.read_sql("ods_info_incr_baostock_earnings_forecast", con=conn.engine)
            ods_stock_base_df = pd.read_sql("ods_info_incr_baostock_stock_base", con=conn.engine)# 获取指数、股票数据
        
        code_dict = dict(zip(ods_stock_base_df['code'], ods_stock_base_df['code_name']))
        earnings_forecast_df['code_name'] = earnings_forecast_df['code'].map(code_dict)
        
        earnings_forecast_df[['profitForcastChgPctUp', 'profitForcastChgPctDwn']] = earnings_forecast_df[['profitForcastChgPctUp', 'profitForcastChgPctDwn']].astype(float).round(1)
        earnings_forecast_df = earnings_forecast_df[(earnings_forecast_df['profitForcastExpPubDate']>='2024-10-01')&
                                                    (earnings_forecast_df['profitForcastChgPctDwn']>=30.)]
        earnings_forecast_df = earnings_forecast_df.sort_values(by='profitForcastChgPctDwn', ascending=False)

        earnings_forecast_df = earnings_forecast_df.rename(columns={
            'code': '证券代码',
            'code_name': '证券名称',
            'profitForcastExpPubDate': '业绩预告发布日期',
            'profitForcastExpStatDate': '业绩预告统计日期',
            'profitForcastType': '业绩预告类型',
            'profitForcastAbstract': '业绩预告摘要',
            'profitForcastChgPctUp':'预告归母净利润增长上限[%]',
            'profitForcastChgPctDwn': '预告归母净利润增长下限[%]',
            })
        earnings_forecast_df = earnings_forecast_df[['证券代码','证券名称','业绩预告发布日期', '业绩预告统计日期', '业绩预告类型', '业绩预告摘要', '预告归母净利润增长上限[%]', '预告归母净利润增长下限[%]']]
        earnings_forecast_df.to_csv(f'{PATH}/data/earnings_forecast.csv', index=False)
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='2024-01-01', help='Start time for backtesting')
    parser.add_argument('--date_end', type=str, default=datetime.today().strftime('%F'), help='End time for backtesting')
    args = parser.parse_args()

    ods_baostock_earnings_forecast_api = odsBaostockEarningsForecastApi()
    ods_baostock_earnings_forecast_api.pipeline(date_start=args.date_start, date_end=args.date_end)
