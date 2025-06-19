# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 22:47:52 2024

@author: awei
adata的财务预报数据(ods_acct_incr_adata_earnings_guidance_api)
"""
import adata
import pandas as pd

from seagull.settings import PATH
from seagull.utils import utils_data, utils_database


def _apply_stock_earnings_guidance_1(subtable):
    stock_code = subtable.name
    ods_adata_stock_earnings_guidance_1 = adata.stock.finance.get_core_index(stock_code=stock_code)
    # ['stock_code', 'report_date', 'dividend_plan', 'ex_dividend_date']
    return ods_adata_stock_earnings_guidance_1


if __name__ == '__main__':
    with utils_database.engine_conn("POSTGRES") as conn:
        ods_adata_stock_base = pd.read_sql("ods_info_incr_adata_stock_base", con=conn.engine)
        
    ods_adata_stock_earnings_guidance = ods_adata_stock_base.groupby('stock_code').apply(_apply_stock_earnings_guidance_1)
    print(ods_adata_stock_earnings_guidance)
    utils_data.output_database(ods_adata_stock_earnings_guidance,
                               filename='ods_acct_incr_adata_earnings_guidance',
                               if_exists='replace') 

# 在这个基础上做环比、同比

# =============================================================================
# stock_code	string	股票代码	
# short_name	string	股票简称	
# report_date	date	报告日期	
# report_type	date	报告类型	
# notice_date	date	公布日期	
# basic_eps	decimal	基本每股收益[元]	
# diluted_eps	decimal	稀释每股收益[元]	
# non_gaap_eps	decimal	扣非每股收益[元]	
# net_asset_ps	decimal	每股净资产[元]	
# cap_reserve_ps	decimal	每股公积金[元]	
# undist_profit_ps	decimal	每股未分配利润[元]	
# oper_cf_ps	decimal	每股经营现金流[元]	
# total_rev	decimal	营业总收入[元]	
# gross_profit	decimal	毛利润[元]	
# net_profit_attr_sh	decimal	归属净利润[元]	
# non_gaap_net_profit	decimal	扣非净利润[元]	
# total_rev_yoy_gr	decimal	营业总收入同比增长[%]	
# net_profit_yoy_gr	decimal	归属净利润同比增长[%]	
# non_gaap_net_profit_yoy_gr	decimal	扣非净利润同比增长[%]	
# total_rev_qoq_gr	decimal	营业总收入滚动环比增长[%]	
# net_profit_qoq_gr	decimal	归属净利润滚动环比增长[%]	
# non_gaap_net_profit_qoq_gr	decimal	扣非净利润滚动环比增长[%]	
# roe_wtd	decimal	净资产收益率[加权][%]	
# roe_non_gaap_wtd	decimal	净资产收益率[扣非/加权][%]	
# roa_wtd	decimal	总资产收益率[加权][%]	
# gross_margin	decimal	毛利率[%]	
# net_margin	decimal	净利率[%]	
# adv_receipts_to_rev	decimal	预收账款/营业总收入	
# net_cf_sales_to_rev	decimal	销售净现金流/营业总收入	
# oper_cf_to_rev	decimal	经营净现金流/营业总收入	
# eff_tax_rate	decimal	实际税率[%]	
# curr_ratio	decimal	流动比率	
# quick_ratio	decimal	速动比率	
# cash_flow_ratio	decimal	现金流量比率	
# asset_liab_ratio	decimal	资产负债率[%]	
# equity_multiplier	decimal	权益系数	
# equity_ratio	decimal	产权比率	
# total_asset_turn_days	decimal	总资产周转天数[天]	
# inv_turn_days	decimal	存货周转天数[天]	
# acct_recv_turn_days	decimal	应收账款周转天数[天]	
# total_asset_turn_rate	decimal	总资产周转率[次]	
# inv_turn_rate	decimal	存货周转率[次]	
# acct_recv_turn_rate	decimal	应收账款周转率[次]
# =============================================================================
