# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 23:04:53 2024

@author: awei
分红数据(ods_acct_incr_adata_dividend_api)
"""
import requests  
from requests.exceptions import JSONDecodeError

import adata
import pandas as pd

from __init__ import path
from utils import utils_data, utils_database


def _apply_stock_dividend_1(subtable):
    stock_code = subtable.name
    try:
        ods_adata_stock_dividend_1 = adata.stock.market.get_dividend(stock_code=stock_code)
        # ['stock_code', 'report_date', 'dividend_plan', 'ex_dividend_date']
        utils_data.output_database(ods_adata_stock_dividend_1,
                                   filename='ods_acct_incr_adata_dividend',
                                   if_exists='append')
        return ods_adata_stock_dividend_1
    except JSONDecodeError:
        print(stock_code)
    except KeyError as e:
        print(e)
    except UnicodeDecodeError as e:
        print(e,'重启IDE')
    except requests.exceptions.RequestException as e:  
    # 处理其他请求错误（如网络问题、超时等）  
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    with utils_database.engine_conn('postgre') as conn:
        ods_adata_stock_base = pd.read_sql("ods_info_incr_adata_stock_base", con=conn.engine)
        
        ods_dividend = pd.read_sql("ods_acct_incr_adata_dividend", con=conn.engine)
        ods_adata_stock_base = ods_adata_stock_base[~(ods_adata_stock_base.stock_code.isin(ods_dividend.stock_code))]

    ods_adata_stock_base.groupby('stock_code').apply(_apply_stock_dividend_1)
    #print(ods_adata_stock_dividend)


# =============================================================================
# stock_code	 string	代码	600001
# report_date	date	公告日	1990-01-01
# dividend_plan	string	分红方案	10股派3.00元，10股转赠5.00股
# ex_dividend_date	date	 除权除息日	1990-01-01
# =============================================================================
