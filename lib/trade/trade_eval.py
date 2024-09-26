# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 10:06:40 2023

@author: awei
交易评估(trade_eval)
"""


def calculate_annualized_return(beginning_value, ending_value, number_of_days):
    # {'2015': 244, '2016': 244, '2017': 244, '2018': 244, '2019': 243 ,'2020': 243 , '2021': 243, '2022': 242}
    # 243Number of trading days closest to many years
    number_of_years = number_of_days / 243
    total_return = (ending_value / beginning_value) - 1
    annualized_return = ((1 + total_return) ** (1 / number_of_years)) - 1
    return annualized_return


if __name__ == '__main__':
    # Example usage:
    beginning_value = 10000  # Initial investment
    ending_value = 15000  # Value after a certain period
    number_of_years = 187  # Number of years
    
    result = calculate_annualized_return(beginning_value, ending_value, number_of_years)
    print(f"Annualized Return: {result * 100:.2f}%")
