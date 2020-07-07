#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
时间窗口划分
ccf_offline_stage1_train.csv: 原始线下训练集信息 -> origin_train_offline_dataset
ccf_online_stage1_train.csv: 原始线上训练集信息 -> origin_train_online_dataset
ccf_offline_stage1_test_revised.csv: 测试集信息 -> test_dataset
得到
"""

import os
import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings
import Fuction_Total as FT

warnings.filterwarnings("ignore")

# 一、数据导入
# os.chdir(r'E:\Data\o2o优惠券使用预测') #设置工作路径
origin_train_offline_dataset = pd.read_csv(r'E:\PycharmProjects\sklearn\Data\o2o_coupon\ccf_offline_stage1_train.csv',
                                           encoding='utf-8')
origin_train_offline_dataset['Date_received'] = pd.to_datetime(origin_train_offline_dataset['Date_received'],
                                                               format='%Y%m%d')
origin_train_offline_dataset['Date'] = pd.to_datetime(origin_train_offline_dataset['Date'], format='%Y%m%d')

origin_train_online_dataset = pd.read_csv(r'E:\PycharmProjects\sklearn\Data\o2o_coupon\ccf_online_stage1_train.csv',
                                          encoding='utf-8')
origin_train_online_dataset['Date_received'] = pd.to_datetime(origin_train_online_dataset['Date_received'],
                                                              format='%Y%m%d')
origin_train_online_dataset['Date'] = pd.to_datetime(origin_train_online_dataset['Date'], format='%Y%m%d')

test_dataset = pd.read_csv(r'E:\PycharmProjects\sklearn\Data\o2o_coupon\ccf_offline_stage1_test_revised.csv',
                           encoding='utf-8')
test_dataset['Date_received'] = pd.to_datetime(test_dataset['Date_received'], format='%Y%m%d')

print('原始线下训练集信息')
print(origin_train_offline_dataset.head(10))
print(origin_train_offline_dataset.info())

print('原始线上训练集信息')
print(origin_train_online_dataset.head(10))
print(origin_train_online_dataset.info())

print('测试集信息')
print(test_dataset.head(10))
print(test_dataset.info())

# 二、根据时间窗口划分训练集和测试集
print('\n--------------------根据时间窗口划分训练集和测试集--------------------\n')
# 线下训练集(特征集)
print('\n线下训练集(特征集)信息：')
# 线下训练集特征train_offline_dataset_feature
# train_offline_dataset_feature：线上领券时间：1.1-4.30，消费时间：1.1-5.15
train_offline_dataset_feature = origin_train_offline_dataset.loc[
    (origin_train_offline_dataset['Date_received'] < pd.Timestamp('2016-05-01')) | (
            (origin_train_offline_dataset['Date_received'].isnull()) & (
            origin_train_offline_dataset['Date'] < pd.Timestamp('2016-05-16')))]
# 对训练集创建目标变量
# 构造训练集线下消费行为变量
# 缺失值替换用于计算(带有时间日期的缺失值无法进行条件筛选,或哪里出错)
train_offline_dataset_feature['Date'].fillna('nan', inplace=True)
train_offline_dataset_feature['Date_received'].fillna('nan', inplace=True)
# 条件筛选(1,2)
train_offline_dataset_feature.loc[train_offline_dataset_feature['Date'] == 'nan', 'off_line_action'] = 1  # 领券但未购买
train_offline_dataset_feature.loc[
    train_offline_dataset_feature['Date_received'] == 'nan', 'off_line_action'] = 2  # 普通消费
# 线下优惠券领取日期和购买日期转换为时间格式(必须放在条件筛选1,2之后，否则新生成的'nan'格式为str,无法构造时间差变量)
train_offline_dataset_feature['Date_received'] = pd.to_datetime(train_offline_dataset_feature['Date_received'])
train_offline_dataset_feature['Date'] = pd.to_datetime(train_offline_dataset_feature['Date'])
# 条件筛选(3,4)
train_offline_dataset_feature['diff_day'] = (
        train_offline_dataset_feature['Date'] - train_offline_dataset_feature['Date_received']).dt.days  # 构造时间差变量
train_offline_dataset_feature.loc[train_offline_dataset_feature['diff_day'] >= 15, 'off_line_action'] = 3  # 领券但15天后消费
train_offline_dataset_feature.loc[train_offline_dataset_feature['diff_day'] < 15, 'off_line_action'] = 4  # 领券且15天内消费
print(train_offline_dataset_feature.head(10))
print(train_offline_dataset_feature.info())

# 线上训练集(特征集)
print('\n线上训练集(特征集)信息：')
# 线上训练集特征train_online_dataset_feature
# train_online_dataset_feature：线上领券时间：1.1-4.30，消费时间：1.1-5.15
train_online_dataset_feature = \
    origin_train_online_dataset.loc \
        [(origin_train_online_dataset['Date_received'] < pd.Timestamp('2016-05-01')) | \
         ((origin_train_online_dataset['Date_received'].isnull()) & (
                 origin_train_online_dataset['Date'] < pd.Timestamp('2016-05-16')))]
print(train_online_dataset_feature.head(10))
print(train_online_dataset_feature.info())

# 训练集(目标集)
# 训练集标签train_dataset_target
print('\n训练集(目标集)信息：')
train_dataset_target = \
    origin_train_offline_dataset.loc \
        [(origin_train_offline_dataset['Date_received'] >= pd.Timestamp('2016-05-16')) &
         (origin_train_offline_dataset['Date_received'] <= pd.Timestamp('2016-06-15'))]
# 对训练集创建目标变量
# (1)构造线下消费行为变量
# 缺失值替换用于计算(带有时间日期的缺失值无法进行条件筛选,或哪里出错)
train_dataset_target['Date'].fillna('nan', inplace=True)
train_dataset_target['Date_received'].fillna('nan', inplace=True)
# 条件筛选(1,2)
train_dataset_target.loc[train_dataset_target['Date'] == 'nan', 'off_line_action'] = 1  # 领券但未购买
train_dataset_target.loc[train_dataset_target['Date_received'] == 'nan', 'off_line_action'] = 2  # 普通消费
# 线下优惠券领取日期和购买日期转换为时间格式(必须放在条件筛选1,2之后，否则新生成的'nan'格式为str,无法构造时间差变量)
train_dataset_target['Date_received'] = pd.to_datetime(train_dataset_target['Date_received'])
train_dataset_target['Date'] = pd.to_datetime(train_dataset_target['Date'])
# 条件筛选(3,4)
train_dataset_target['diff_day'] = (
        train_dataset_target['Date'] - train_dataset_target['Date_received']).dt.days  # 构造时间差变量
train_dataset_target.loc[train_dataset_target['diff_day'] >= 15, 'off_line_action'] = 3  # 领券但15天后消费
train_dataset_target.loc[train_dataset_target['diff_day'] < 15, 'off_line_action'] = 4  # 领券且15天内消费
# (2)构造目标变量
train_dataset_target['target'] = np.where(train_dataset_target['off_line_action'] == 4, 1, 0)
# 删除辅助变量
train_dataset_target.drop(['off_line_action', 'diff_day'], axis=1, inplace=True)

print(train_dataset_target.head(10))
print(train_dataset_target.info())

'''#注：
#train_dataset_target中不同的user_id共14W，其中6W在train_offline_dataset_feature中不存在
#train_dataset_target中不同的Merchant_id共4300种，其中700种在train_offline_dataset_feature中不存在'''
# 1、删除train_dataset_target中在train_offline_dataset_feature中User_id不同的记录
unique_user_id_train_feature = train_offline_dataset_feature['User_id'].unique()
unique_user_id_train_target = train_dataset_target['User_id'].unique()
# (1)unique_user_id_train_feature与unique_user_id_train_target的交集
common_list = list(set(unique_user_id_train_target) & set(unique_user_id_train_feature))
# (2)返回只在unique_user_id_train_target中有,而在交集中没有的元素唯一值
unique_list = np.setdiff1d(unique_user_id_train_target, common_list)
# (3)找出特征集中存在但目标集中不存在的user_id的index
drop_index_unique_user = train_dataset_target.loc[train_dataset_target['User_id'].isin(unique_list)].index
# (4)从特征集中删除
train_dataset_target.drop(drop_index_unique_user, axis=0, inplace=True)

print('\n删除多余用户id后的训练集(目标集)信息')
print(train_dataset_target.head(10))
print(train_dataset_target.info())

# 2、删除train_dataset_target中在train_offline_dataset_feature中Merchant_id不同的记录
unique_Merchant_id_train_feature = train_offline_dataset_feature['Merchant_id'].unique()
unique_Merchant_id_train_target = train_dataset_target['Merchant_id'].unique()
# (1)unique_Merchant_id_train_feature与unique_Merchant_id_train_target的交集
common_list = list(set(unique_Merchant_id_train_feature) & set(unique_Merchant_id_train_target))
# (2)返回只在unique_Merchant_id_train_target中有,而在交集中没有的元素唯一值
unique_list = np.setdiff1d(unique_Merchant_id_train_target, common_list)
# (3)找出特征集中存在但目标集中不存在的user_id的index
drop_index_unique_Merchant = train_dataset_target.loc[train_dataset_target['Merchant_id'].isin(unique_list)].index
# (4)从特征集中删除
train_dataset_target.drop(drop_index_unique_Merchant, axis=0, inplace=True)

print('\n删除多余商户id后的训练集(目标集)信息')
print(train_dataset_target.head(10))
print(train_dataset_target.info())

# 生成新的训练集(线下特征集，线上特征集，目标集)
train_offline_dataset_feature.to_csv(r'E:\PycharmProjects\sklearn\Data\o2o_coupon\train_offline_dataset_feature.csv',
                                     index=False)
train_online_dataset_feature.to_csv(r'E:\PycharmProjects\sklearn\Data\o2o_coupon\train_online_dataset_feature.csv',
                                    index=False)
train_dataset_target.to_csv(r'E:\PycharmProjects\sklearn\Data\o2o_coupon\train_dataset_target.csv', index=False)

# 线下测试集(特征集)
print('\n线下测试集(特征集)信息：')
# 线下测试集特征test_offline_dataset_feature
# test_offline_dataset_feature：线下领券时间或消费时间在03.01-06.30之间
test_offline_dataset_feature = \
    origin_train_offline_dataset.loc \
        [(origin_train_offline_dataset['Date_received'].dt.month >= 3) | (
            origin_train_offline_dataset['Date'].dt.month >= 3)]
# test_offline_dataset_feature2：test_offline_dataset_feature中领券时间<3月的
test_offline_dataset_feature2 = test_offline_dataset_feature.loc[
    test_offline_dataset_feature['Date_received'].dt.month <= 2]
# test_offline_dataset_feature(删除不符合的行后完整的特征集)
drop_index_offline = test_offline_dataset_feature2.index
test_offline_dataset_feature.drop(drop_index_offline, axis=0, inplace=True)
'''漏洞，上述筛选会造成519行Date_received<3但是Date>=3的样本出现在线下测试集(已解决)
print(test_offline_dataset_feature.loc[test_offline_dataset_feature['Date_received'].dt.month==2])
print(test_offline_dataset_feature.loc[test_offline_dataset_feature['Date_received'].dt.month==2].info())'''
# 构造测试集线下消费行为变量
# 缺失值替换用于计算(带有时间日期的缺失值无法进行条件筛选,或哪里出错)
test_offline_dataset_feature['Date'].fillna('nan', inplace=True)
test_offline_dataset_feature['Date_received'].fillna('nan', inplace=True)
# 条件筛选(1,2)
test_offline_dataset_feature.loc[test_offline_dataset_feature['Date'] == 'nan', 'off_line_action'] = 1  # 领券但未购买
test_offline_dataset_feature.loc[test_offline_dataset_feature['Date_received'] == 'nan', 'off_line_action'] = 2  # 普通消费
# 线下优惠券领取日期和购买日期转换为时间格式(必须放在条件筛选1,2之后，否则新生成的'nan'格式为str,无法构造时间差变量)
test_offline_dataset_feature['Date_received'] = pd.to_datetime(test_offline_dataset_feature['Date_received'])
test_offline_dataset_feature['Date'] = pd.to_datetime(test_offline_dataset_feature['Date'])
# 条件筛选(3,4)
test_offline_dataset_feature['diff_day'] = \
    (test_offline_dataset_feature['Date'] - test_offline_dataset_feature['Date_received']).dt.days  # 构造时间差变量
test_offline_dataset_feature.loc[test_offline_dataset_feature['diff_day'] >= 15, 'off_line_action'] = 3  # 领券但15天后消费
test_offline_dataset_feature.loc[test_offline_dataset_feature['diff_day'] < 15, 'off_line_action'] = 4  # 领券且15天内消费
print(test_offline_dataset_feature.head(10))
print(test_offline_dataset_feature.info())

# 线上测试集(特征集)
print('\n线上测试集(特征集)信息：')
# 线上测试集特征test_online_dataset_feature
# test_online_dataset_feature：线上领券时间或消费时间在03.01-06.30之间
'''问题同线下测试集（特征集）(已解决)'''
test_online_dataset_feature = \
    origin_train_online_dataset.loc \
        [(origin_train_online_dataset['Date_received'].dt.month >= 3) | (
            origin_train_online_dataset['Date'].dt.month >= 3)]
# test_online_dataset_feature2：test_online_dataset_feature中领券时间<3月的
test_online_dataset_feature2 = test_online_dataset_feature.loc[
    test_online_dataset_feature['Date_received'].dt.month <= 2]
# test_online_dataset_feature(删除不符合的行后完整的特征集)
drop_index_online = test_online_dataset_feature2.index
test_online_dataset_feature.drop(drop_index_online, axis=0, inplace=True)

print(test_online_dataset_feature.head(10))
print(test_online_dataset_feature.info())

# 生成新的测试集(线下特征集，线上特征集，目标集)
test_offline_dataset_feature.to_csv(r'E:\PycharmProjects\sklearn\Data\o2o_coupon\test_offline_dataset_feature.csv',
                                    index=False)
test_online_dataset_feature.to_csv(r'E:\PycharmProjects\sklearn\Data\o2o_coupon\test_online_dataset_feature.csv',
                                   index=False)
test_dataset.to_csv(r'E:\PycharmProjects\sklearn\Data\o2o_coupon\test_dataset_target.csv', index=False)

# 注：
# test_dataset_target中不同的user_id共7.6W，其0.9W中test_offline_dataset_feature中不存在
# test_dataset_target中不同的Merchant_id共1559种，其中3种在test_offline_dataset_feature中不存在


print("DONE")
