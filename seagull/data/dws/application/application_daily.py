# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 10:58:16 2023

@author: awei
应用层_每日定时任务(application_daily)
"""
import argparse
import schedule
from loguru import logger

import baostock as bs

from seagull.settings import PATH
from seagull.data import data_ods_a_stock_k
#from data.ods import odsData
#ods_data = odsData()
from seagull.data import data_ods_ohlc_incr_baostock_stock_sh_sz_cycle
def custom_date(date=None):
    """
    功能：Directly execute daily tasks for the day
    """
    try:
        bs.login()
        
        ## ODS层
        #ods_data.ods_baostock(data_type='交易日')
        #ods_data.ods_baostock(data_type='行业分类')
        #ods_data.ods_baostock(data_type='证券资料')
        #ods_data.ods_baostock(data_type='证券代码')
        #ods_data.ods_adata_portfolio_base(data_type='ETF代码')
        #ods_data.ods_efinance_portfolio(data_type='ETF日频')
        # get_data.api_efinance(data_type='ETF五分钟频')
        
        # 每日k线
        ods_incr_baostock_stock_sh_sz_cycle = data_ods_ohlc_incr_baostock_stock_sh_sz_cycle.OdsIncrBaostockStockShSzCycle()
        stock_sh_sz_daily_df = ods_incr_baostock_stock_sh_sz_cycle.stock_sh_sz_daily(date_start=args.date_start, date_end=args.date_end)
        
#ods_ohlc_incr_efinance_stock_bj_api
    ods_efinance_stock_bj_api = odsEfinanceStockBjApi()
    date_end = datetime.now().strftime("%F")
    if args.update_type=='full':
        ods_efinance_stock_bj_api.stock_bj_daily(date_end=date_end)
    elif args.update_type=='incr':
        date_start = utils_data.maximum_date_next(table_name='ods_ohlc_incr_efinance_stock_bj_daily', field_name='日期')
        ods_efinance_stock_bj_api.stock_bj_daily(date_start=date_start, date_end=date_end)        
        ## DWD层
        
        
        # ods_ohlc_incr_efinance_stock_portfolio_api
        ods_efinance_stock_portfolio_api = odsEfinanceStockPortfolioApi()
        date_end = args.date_end if args.date_end!='' else datetime.now().strftime("%F")
        if args.update_type=='full':
            ods_efinance_stock_portfolio_api.stock_daily(date_end=date_end)
        elif args.update_type=='incr':
            date_start = utils_data.maximum_date_next(table_name='ods_ohlc_incr_efinance_portfolio_daily', field_name='日期')
            ods_efinance_stock_portfolio_api.stock_daily(date_start=date_start, date_end=date_end)
        
        
        # dwd_ohlc_incr_stock
        
        #dwd_feat_incr_macd
        #ods_feat_incr_adata_capital_flow_api  --  dwd_feat_incr_capital_flow
        #feature.alpha101  --  dwd_feat_incr_alpha
        
        
        #dwd_feat_incr_global_index
        #dwd_feat_incr_indicators
        
        #lightgbm_data
    except Exception as e:
        logger.error(f"Exception when logging in to obtain transaction day data: {e}")
    finally:
        bs.logout()

def time():
    """
    功能：任务定时
    """
    schedule.every().day.at("17:18").do(custom_date)
    while True:
        schedule.run_pending()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default='now')  # 默认daily每日离线任务,可选['daily','now','2022-09-13']
    parser.add_argument('--update_type', type=str, default='incr', help='Data update method')
    args = parser.parse_args()
    
    if args.date not in ['daily', 'now']:
        custom_date(args.date)
    elif args.date == 'now':
        custom_date()
    elif args.date == 'daily':
        time()



