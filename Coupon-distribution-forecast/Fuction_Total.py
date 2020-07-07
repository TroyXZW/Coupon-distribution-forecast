#!/usr/bin/env python 
# -*- coding:utf-8 -*-

"""
此脚本用于定义函数，供其他脚本调用
"""


def Disc_Cal(x):
    """
    定义折扣计算函数
    """

    if ':' in str(x):
        front = float(str(x)[0:str(x).rfind(':')])
        behind = float(str(x)[str(x).rfind(':') + 1:])
        return (front - behind) / front
    else:
        return float(x)


def Lowest_Disc_Cal(x):
    """
    定义最低折扣消费计算函数
    """

    if ':' in str(x):
        min_buy = float(str(x)[0:str(x).rfind(':')])
        return min_buy
    elif '.' in str(x):
        return float(0)
    else:
        return x


def drop_duplicate(x, i):
    """
    剔除重复值程序(如果有的话)
    """

    if ((x.value_counts().sort_values(ascending=False).max() / x.value_counts().sum()) > 0.95):
        print('去除的特征为{0}，该特征内重复值占比为{1:.4f}\n'.format(i, (x.value_counts().sort_values(ascending=False).max() /
                                                         x.value_counts().sum())))
        del x


def to_int(x):
    """
    时间变量转化为int类型
    定义转化函数
    """

    if '-' in str(x):
        x = str(x).replace(x[4], '')
        return int(x)
    else:
        return int(x)

    
print("DONE")
