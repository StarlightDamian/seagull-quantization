# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 19:02:41 2025

@author: Damian

pipeline
"""



ods_info_full_asset_base_details
adata的股票所属板块、概念(ods_flag_full_adata_label)

获取所有场内ETF当前信息(ods_info_nrtd_adata_portfolio_base_reptile)
ods_info_incr_efinance_trading_day.py
ods_info_incr_adata_stock_base


获取ods层基本信息(data_ods_info_incr_baostock_stock_base_api)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
   # parser.add_argument('--date_start', type=str, default='2019-01-01', help='Start time for backtesting')
    #parser.add_argument('--date_start', type=str, default='2024-01-26', help='Start time for backtesting')
    parser.add_argument('--date_start', type=str, default='1990-01-01', help='Start time for backtesting')
    parser.add_argument('--date_end', type=str, default='2025-05-22', help='End time for backtesting')
    args = parser.parse_args()
    

    ods_incr_baostock_stock_sh_sz_cycle = OdsIncrBaostockStockShSzCycle()






dwd

ods_ohlc_incr_efinance_stock_portfolio_api
ods_ohlc_incr_efinance_stock_bj_api

ods_info_incr_efinance_trading_day


ods_info_incr_baostock_trade_stock
data_dwd_info_incr_stock_base
dwd_ohlc_incr_stock



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='1990-01-01', help='Start time for backtesting')
    parser.add_argument('--date_end', type=str, default='', help='End time for backtesting')
    parser.add_argument('--update_type', type=str, default='incr', help='Data update method')
    parser.add_argument('--filename', type=str, default='dwd_feat_incr_alpha', help='Database table name')
    args = parser.parse_args()
    
    date_end = args.date_end if args.date_end!='' else datetime.now().strftime("%F")
    if args.update_type=='full':
        df = pipeline(date_start=args.date_start,
                      date_end=date_end)
        utils_data.output_database_large(df,
                                         filename=args.filename,
                                         if_exists='replace')
    elif args.update_type=='incr':
        date_start = utils_data.maximum_date_next(table_name=args.filename)
        trading_day_alignment = finance_trading_day.TradingDayAlignment()
        #date_start='2023-01-01'
        date_start_prev = trading_day_alignment.shift_day(date_start=date_start, date_num=100)
        
        #raw_df = pipeline(date_start=date_start_prev, date_end=date_end)
        with utils_database.engine_conn('postgre') as conn:
            df = pd.read_sql(f"""SELECT
                                     * 
                                 FROM
                                     dwd_ohlc_incr_stock_daily
                                 WHERE
                                     date BETWEEN '{date_start}' AND '{date_end}'
                                     """, con=conn.engine)
        df = df.drop_duplicates('primary_key', keep='first') # 不去重会导致数据混入测试集
        df = df.sort_values(by='date').reset_index(drop=True)
        
        #grouped = df.groupby('full_code')
        #alpha_df = utils_thread.thread(grouped, get_alpha, max_workers=6)
        alpha_df = df.groupby('full_code').apply(get_alpha)
        alpha_columns = [x for x in alpha_df.columns if 'alpha' in x]
        raw_df = alpha_df[['primary_key', 'date', 'time', 'board_type', 'full_code','freq','adj_type'] + alpha_columns]

        df = raw_df[raw_df.date>=date_start]
        utils_data.output_database_large(df,
                                         filename=args.filename,
                                         if_exists='append')
        logger.info('finsh')
        
        
        ods_flag_full_baostock_stock_label
        dwd_flag_full_label
    lightgbm_data