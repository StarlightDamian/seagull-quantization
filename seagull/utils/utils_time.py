# -*- coding: utf-8 -*-
"""
@Date: 2022/5/16 13:07
@Author: Damian
@Email: zengyuwei1995@163.com
@File: utils_time.py
@Description: 时间
"""
from datetime import datetime, timedelta,date

import pandas as pd


def __gen_dates(b_date, days):
    day = timedelta(days=1)
    for i in range(days.days):
        yield b_date + day*i


def __date_list(start, end):
    start_datetime = datetime.strptime(start, '%Y-%m-%d')
    end_datetime = datetime.strptime(end, '%Y-%m-%d')
    end_datetime = (end_datetime+timedelta(days=1))  # 结尾要加一天
    data = []
    for day in __gen_dates(start_datetime, (end_datetime - start_datetime)):
        data.append(day)
    return data


def date_binary_list(start_date, end_date):
    date_datetime = __date_list(start_date, end_date)
    date_list = [str(x)[:10] for x in date_datetime]
    date_binary_pair_list = []
    for idx, date in enumerate(date_list):
        try:
            date_binary_pair_list.append([date, date_list[idx+1]])
        except:
            break
    return date_binary_pair_list


def date_replace_binary_replace_list(date_start_replace, date_end_replace):
    date_start = datetime.strptime(date_start_replace, '%Y%m%d').strftime('%F')
    date_end = datetime.strptime(date_end_replace, '%Y%m%d').strftime('%F')
    return date_binary_replace_list(date_start, date_end)


def date_binary_replace_list(date_start, date_end):
    date_start = datetime.strptime(date_start, '%Y%m%d').strftime('%F') if '-' not in date_start else date_start
    date_end = datetime.strptime(date_end, '%Y%m%d').strftime('%F') if '-' not in date_end else date_end
    date_binary_pair_list = date_binary_list(date_start, date_end)
    date_binary_replace_list = [[x[0].replace('-', ''), x[1].replace('-', '')] for x in date_binary_pair_list]
    return date_binary_replace_list


def run_many_days(date_start, date_end, func):
    """
    功能：按日期跑每日数据，然后拼接.既可以跑一天的数据，也可以兼容多日数据合并的需求
    输入：date_start：'2022-06-13'
        func：输出单日数据的函数块
    输出：data_many_days_list:每日数据组成的列表
    """
    data_many_days_list = [func(date_start_replace, date_end_replace) for (date_start_replace, date_end_replace) in date_binary_replace_list(date_start, date_end)]
    return data_many_days_list


def date_suffix(date_type='today'):
    """
    功能：获取今日日期后缀
    备注：如今天是2022年2月23日。则输出 _20220223_20220224
    """
    today = datetime.now().strftime('%Y%m%d')
    if date_type == 'today':
        tomorrow = (datetime.now()+timedelta(days=1)).strftime('%Y%m%d')
        return '_' + today + '_' + tomorrow
    elif date_type == 'yesterday':
        yesterday = (datetime.now()+timedelta(days=-1)).strftime('%Y%m%d')
        return '_' + yesterday + '_' + today


def today_date_range(date_type='today'):
    """
    功能：获取今日日期列表
    备注：如今天是2022年2月23日。则输出 ['2022-02-23', '2022-02-24']
    """
    today = datetime.now().strftime('%F')
    if date_type == 'today':
        tomorrow = (datetime.now()+timedelta(days=1)).strftime('%F')
        return [today, tomorrow]
    elif date_type == 'yesterday':
        yesterday = (datetime.now()+timedelta(days=-1)).strftime('%F')
        return [yesterday, today]


def time_range_h(date_start='2024-03-08', date_end='2024-05-24'):
    hourly_range = pd.date_range(start=date_start, end=date_end, freq='H')
    time_pd = pd.DataFrame(hourly_range, columns=['time_start'])
    time_pd['time_end'] = time_pd['time_start'] + pd.Timedelta(minutes=59, seconds=59)
    time_pd['time_str'] = time_pd['time_start'].dt.strftime('%Y%m%d%H%M%S')+'_'+time_pd['time_end'].dt.strftime('%Y%m%d%H%M%S')
    return time_pd


def date_plus_days(date_start="2000-01-01", days=1):
    start_date = datetime.strptime(date_start, "%Y-%m-%d")
    date_n_days_later = start_date + timedelta(days=days)
    formatted_date = date_n_days_later.strftime('%Y-%m-%d')
    return formatted_date  # '2000-04-11'


import itertools
from typing import List, Dict, Any


def split_time_windows(
        start_date: str,
        end_date: str,
        window_days: int
) -> List[Dict[str, str]]:
    """
    将 [start_date, end_date] 区间按 window_days 切分，返回一系列 {'date_start', 'date_end'} dict。
    自动将 end_date 限制为不超过今天。
    """
    today = date.today().isoformat()
    # 取不超过今天的结束日期
    end_date = min(end_date, today)

    windows = []
    current_start = start_date
    while current_start < end_date:
        # 下一个结束点
        next_end = date_plus_days(current_start, days=window_days)
        current_end = next_end if next_end < end_date else end_date

        windows.append({
            'date_start': current_start,
            'date_end': current_end
        })

        current_start = current_end

    return windows


def make_param_grid(
        date_start: str,
        date_end: str,
        window_days: int = 30,
        **dimensions: List[Any]
) -> List[Dict[str, Any]]:
    """
    生成参数字典列表：
      - 先用 split_time_windows 切出所有时间窗口；
      - 再对传入的每个 dimension（如 full_code=[...], users=[...]）做笛卡尔积，
        将每个组合与每个时间窗口逐一合并。

    :param date_start:   全局起始日期 "YYYY-MM-DD"
    :param date_end:     全局结束日期上限 "YYYY-MM-DD"
    :param window_days:  每个子区间天数，默认 30
    :param dimensions:   可选的命名维度列表，如 full_code=['sh.000001', ...]、users=['u1','u2'] 等
    :return: List of dicts，key 包含 'date_start','date_end' + 所有维度名
    """
    # 切分出的所有时间窗口
    time_windows = split_time_windows(date_start, date_end, window_days)

    # 如果没有额外维度，只返回时间窗口
    if not dimensions:
        return time_windows

    # 准备笛卡尔积：dim_names=['full_code','users',...], dim_values=[ [...], [...] ]
    dim_names = list(dimensions.keys())
    dim_values = [dimensions[k] for k in dim_names]

    param_list = []
    # 对每个维度组合 × 每个时间窗口
    for combo in itertools.product(*dim_values):
        base = dict(zip(dim_names, combo))
        for tw in time_windows:
            entry = {**base, **tw}
            param_list.append(entry)

    return param_list


if __name__ == '__main__':
    print(date_binary_list('2022-04-03', '2022-04-07'))
    print(date_binary_replace_list('20220403', '20220407'))
    
