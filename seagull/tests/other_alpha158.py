# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:20:33 2024

@author: awei
"""

import qlib
from qlib.contrib.data.handler import Alpha158

data_handler_config = {
    "start_time": "2008-01-01",
    "end_time": "2020-08-01",
    "fit_start_time": "2008-01-01",
    "fit_end_time": "2014-12-31",
    "instruments": "csi300",
}

if __name__ == "__main__":
    """
    下载并准备数据：执行以下命令下载股票数据。这些数据是从雅虎财经收集的，
    建议用户准备自己的数据
    """
    qlib.init()
    h = Alpha158(**data_handler_config)

    # get all the columns of the data
    print(h.get_cols())

    # fetch all the labels
    print(h.fetch(col_set="label"))

    # fetch all the features
    print(h.fetch(col_set="feature"))