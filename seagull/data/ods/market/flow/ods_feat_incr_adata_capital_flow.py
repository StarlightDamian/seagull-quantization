# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 23:52:09 2024

@author: awei
(ods_feat_incr_adata_capital_flow_api)

没必要每天刷新
adata的资金流动接口很多股票没有对应数据
"""
import requests  
from requests.exceptions import JSONDecodeError

import adata
import pandas as pd

from seagull.settings import PATH
from seagull.utils import utils_data, utils_database, utils_thread


def _apply_stock_capital_flow_1(subtable):
    #stock_code = subtable.name
    stock_code = subtable.stock_code.values[0]
    print(stock_code,'start')
    try:
        ods_adata_capital_flow_1 = adata.stock.market.get_capital_flow(stock_code=stock_code,
                                                                       #start_date='2020-01-01'
                                                                       )
    # ['stock_code', 'report_date', 'dividend_plan', 'ex_dividend_date']
        print(ods_adata_capital_flow_1)
        utils_data.output_database(ods_adata_capital_flow_1,
                                   filename='ods_feat_incr_adata_capital_flow',
                                   if_exists='append')
        #return ods_adata_capital_flow_1
    except JSONDecodeError:
        print(stock_code)
    except KeyError as e:
        print(e)
    except UnicodeDecodeError as e:
        print(e,'重启IDE')
    except TypeError as e:
        print(e,'解析异常')
    except requests.exceptions.RequestException as e:  
    # 处理其他请求错误（如网络问题、超时等）  
        print(e)


if __name__ == '__main__':
    with utils_database.engine_conn("POSTGRES") as conn:
        ods_adata_stock_base = pd.read_sql("ods_info_incr_adata_stock_base", con=conn.engine)
        
        ods_capital_flow = pd.read_sql("ods_feat_incr_adata_capital_flow", con=conn.engine)
        ods_adata_stock_base = ods_adata_stock_base[~(ods_adata_stock_base.stock_code.isin(ods_capital_flow.stock_code))]
    
    #grouped = ods_adata_stock_base.groupby('stock_code')
    #utils_thread.thread(grouped, apply_stock_capital_flow_1, max_workers=8)
    ods_adata_stock_base.groupby('stock_code').apply(_apply_stock_capital_flow_1)
    #print(ods_adata_capital_flow)


# =============================================================================
#  {'closepx': '11.73',
#   'date': '2024/12/11',
#   'extMainIn': '-1.14亿',
#   'largeNetIn': '+5819.53万',
#   'littleNetIn': '+8701.17万',
#   'mediumNetIn': '+2736.44万',
#   'ratio': '-0.51%',
#   'showtime': '2024-12-11',
#   'superNetIn': '-1.73亿',
#   'time': '20241211'},
#  {'closepx': '11.79',
#   'date': '2024/12/10',
#   'extMainIn': '+1.80亿',
#   'largeNetIn': '+1.02亿',
#   'littleNetIn': '-4623.89万',
#   'mediumNetIn': '-1.34亿',
#   'ratio': '+1.03%',
#   'showtime': '2024-12-10',
#   'superNetIn': '+7849.92万',
#   'time': '20241210'},
#  {'closepx': '11.67',
#   'date': '2024/12/09',
#   'extMainIn': '-1923.72万',
#   'largeNetIn': '+1719.26万',
#   'littleNetIn': '-227.84万',
#   'mediumNetIn': '+2151.56万',
#   'ratio': '+0.09%',
#   'showtime': '2024-12-09',
#   'superNetIn': '-3642.98万',
#   'time': '20241209'},
#  {'closepx': '11.66',
#   'date': '2024/12/06',
#   'extMainIn': '+2.95亿',
#   'largeNetIn': '+780.10万',
#   'littleNetIn': '-9818.70万',
#   'mediumNetIn': '-1.97亿',
#   'ratio': '+1.92%',
#   'showtime': '2024-12-06',
#   'superNetIn': '+2.87亿',
#   'time': '20241206'},
#  {'closepx': '11.44',
#   'date': '2024/12/05',
#   'extMainIn': '-1286.86万',
#   'largeNetIn': '+4809.78万',
#   'littleNetIn': '+1387.22万',
#   'mediumNetIn': '-100.36万',
#   'ratio': '-0.17%',
#   'showtime': '2024-12-05',
#   'superNetIn': '-6096.64万',
#   'time': '20241205'},
#  {'closepx': '11.46',
#   'date': '2024/12/04',
#   'extMainIn': '-4527.96万',
#   'largeNetIn': '+2466.81万',
#   'littleNetIn': '+4155.44万',
#   'mediumNetIn': '+372.53万',
#   'ratio': '-0.26%',
#   'showtime': '2024-12-04',
#   'superNetIn': '-6994.77万',
#   'time': '20241204'}]
# =============================================================================
