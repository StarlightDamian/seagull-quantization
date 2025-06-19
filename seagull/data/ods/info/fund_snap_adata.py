# -*- coding: utf-8 -*-
"""
@Date: 2024/8/7 13:07
@Author: Damian
@Email: zengyuwei1995@163.com
@File: fund_snap_adata.py
@Description:
获取所有场内ETF当前信息(ods/info/fund_snap_adata)
重写adata的adata.fund.info.all_etf_exchange_traded_info()接口，额外新增获取ETF的市场名称
因为是爬虫数据，所以无法获取到历史的etf数据
net_value净值数据是增量数据，15秒更新一次
"""
import json
import requests

import pandas as pd
from adata.fund.info.fund_info import FundInfo

from seagull.settings import PATH
from seagull.utils import utils_data


class odsNrtdAdataPortfolioBaseReptile(FundInfo):
    """
    ETF信息,只有SH和SZ的ETF
    """
    __ETF_INFO_COLUMNS = ['market_code', 'fund_code', 'short_name', 'net_value']
    
    def __init__(self) -> None:
        super().__init__()
        
    def all_etf_exchange_traded_info_east(self, wait_time=None):
        """
        http://68.push2.eastmoney.com/api/qt/clist/get?cb=jQuery1124047482019788167995_1690884441114&pn=1&pz=500&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&wbp2u=|0|0|0|web&fid=f3&fs=b:MK0021,b:MK0022,b:MK0023,b:MK0024&fields=f12,f14,f2&_=1690884441121
        :param wait_time: 等待时间
        :return:
        """
        curr_page = 1
        data = []
        while curr_page < 5:
            url = f"http://68.push2.eastmoney.com/api/qt/clist/get?cb=jQuery1124047482019788167995_1690884441114" \
                  f"&pn={curr_page}&pz=500&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&wbp2u=|0|0|0|web" \
                  f"&fid=f3&fs=b:MK0021,b:MK0022,b:MK0023,b:MK0024&fields=f12,f14,f2,f13&_=1690884441121"
            text = requests.request('get', url, headers={}, proxies={}).text
            res_json = json.loads(text[text.index('{'):-2])
            res_data = res_json['data']
            if not res_data:
                break
            res_data = res_data['diff']
            for _ in res_data:
                data.append({'fund_code': _['f12'],
                             'short_name': _['f14'],
                             'net_value': _['f2'],
                             'market_code': _['f13']})
            curr_page += 1
        result_df = pd.DataFrame(data=data, columns=self.__ETF_INFO_COLUMNS)
        return result_df

    def portfolio_base(self):
        portfolio_base_df = self.all_etf_exchange_traded_info_east()
        utils_data.output_database(portfolio_base_df,
                                   filename='ods_info_nrtd_adata_portfolio_base',
                                   if_exists='replace')


if __name__ == '__main__':
    ods_snap_adata_portfolio_base_reptile = odsNrtdAdataPortfolioBaseReptile()
    ods_snap_adata_portfolio_base_reptile.portfolio_base()
