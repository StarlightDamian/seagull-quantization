# -*- coding: utf-8 -*-
"""
@Date: 2022/5/16 13:07
@Author: Damian
@Email: zengyuwei1995@163.com
@File: utils_time.py
@Description: 时间
"""
from datetime import datetime, timedelta

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


def date_plus_days(date_start = "2000-01-01", days=1):
    start_date = datetime.strptime(date_start, "%Y-%m-%d")
    date_n_days_later = start_date + timedelta(days=days)
    formatted_date = date_n_days_later.strftime('%Y-%m-%d')
    return formatted_date  # '2000-04-11'


if __name__ == '__main__':
    print(date_binary_list('2022-04-03', '2022-04-07'))
    print(date_binary_replace_list('20220403', '20220407'))
    
