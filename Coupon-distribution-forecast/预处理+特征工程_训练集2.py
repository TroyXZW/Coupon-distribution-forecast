#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
缺失值处理、重复值处理、异常值检验
特征工程(对训练集和测试集同时操作,每个变量衍生步骤：训练集衍生-缺失值填充-对应测试集衍生-测试集缺失值填充)
生成origin_train_dataset.csv
"""

import os
import re
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings
import Fuction_Total as FT

warnings.filterwarnings("ignore")

# 数据导入
# 1、训练集
# os.chdir(r'E:\Data\o2o优惠券使用预测')  # 设置工作路径
train_offline_dataset_feature = pd.read_csv(
    r'E:\PycharmProjects\sklearn\Data\o2o_coupon\train_offline_dataset_feature.csv', encoding='utf-8')
train_offline_dataset_feature['Date_received'] = pd.to_datetime(train_offline_dataset_feature['Date_received'])
train_offline_dataset_feature['Date'] = pd.to_datetime(train_offline_dataset_feature['Date'])

train_online_dataset_feature = pd.read_csv(
    r'E:\PycharmProjects\sklearn\Data\o2o_coupon\train_online_dataset_feature.csv', encoding='utf-8')
train_online_dataset_feature['Date_received'] = pd.to_datetime(train_online_dataset_feature['Date_received'])
train_online_dataset_feature['Date'] = pd.to_datetime(train_online_dataset_feature['Date'])

train_dataset_target = pd.read_csv(r'E:\PycharmProjects\sklearn\Data\o2o_coupon\train_dataset_target.csv',
                                   encoding='utf-8')
train_dataset_target['Date_received'] = pd.to_datetime(train_dataset_target['Date_received'])

print('\n线下训练集(特征集)信息')
print(train_offline_dataset_feature.head(10))
print(train_offline_dataset_feature.info())

print('\n线上训练集(特征集)信息')
print(train_online_dataset_feature.head(10))
print(train_online_dataset_feature.info())

print('\n训练集(目标集)信息')
print(train_dataset_target.head(10))
print(train_dataset_target.info())

# 2、测试集
test_offline_dataset_feature = pd.read_csv(
    r'E:\PycharmProjects\sklearn\Data\o2o_coupon\test_offline_dataset_feature.csv', encoding='utf-8')
test_offline_dataset_feature['Date_received'] = pd.to_datetime(test_offline_dataset_feature['Date_received'])
test_offline_dataset_feature['Date'] = pd.to_datetime(test_offline_dataset_feature['Date'])

test_online_dataset_feature = pd.read_csv(r'E:\PycharmProjects\sklearn\Data\o2o_coupon\test_online_dataset_feature.csv',
                                          encoding='utf-8')
test_online_dataset_feature['Date_received'] = pd.to_datetime(test_online_dataset_feature['Date_received'])
test_online_dataset_feature['Date'] = pd.to_datetime(test_online_dataset_feature['Date'])

test_dataset_target = pd.read_csv(r'E:\PycharmProjects\sklearn\Data\o2o_coupon\test_dataset_target.csv',
                                  encoding='utf-8')
test_dataset_target['Date_received'] = pd.to_datetime(test_dataset_target['Date_received'])

print('\n线下测试集(特征集)信息')
print(test_offline_dataset_feature.head(10))
print(test_offline_dataset_feature.info())

print('\n线上测试集(特征集)信息')
print(test_online_dataset_feature.head(10))
print(test_online_dataset_feature.info())

print('\n测试集(目标集)信息')
print(test_dataset_target.head(10))
print(test_dataset_target.info())

# 三、数据预处理(同时对训练集和测试集的特征集进行)
print('\n--------------------数据预处理--------------------\n')
# 1、缺失值处理
# 检测
# (1)线下训练集(特征集)
print('线下训练集缺失值检测：')
null_or_not_train_offline = train_offline_dataset_feature.apply(lambda x: sum(x.isnull() / len(x)), axis=0)
print(null_or_not_train_offline)

# (2)线上训练集集(特征集)
print('\n线上训练集缺失值检测：')
null_or_not_train_online = train_online_dataset_feature.apply(lambda x: sum(x.isnull() / len(x)), axis=0)
print(null_or_not_train_online)

# (3)线下测试集(特征集)
print('\n线下测试集(特征集)缺失值检测：')
null_or_not_test_offline = test_offline_dataset_feature.apply(lambda x: sum(x.isnull() / len(x)), axis=0)
print(null_or_not_test_offline)

# (4)线上测试集(特征集)
print('\n线上测试集(特征集)缺失值检测：')
null_or_not_test_online = test_online_dataset_feature.apply(lambda x: sum(x.isnull() / len(x)), axis=0)
print(null_or_not_test_online)

# 线下训练集(特征集)Distance用平均数(取整)填充
train_offline_dataset_feature['Distance'].fillna(round(train_offline_dataset_feature['Distance'].mean()), inplace=True)
# 线下测试集(特征集)Distance用平均数(取整)填充
test_offline_dataset_feature['Distance'].fillna(round(test_offline_dataset_feature['Distance'].mean()), inplace=True)

# 2、重复值检测
# (检测每列中重复字段所占比例)
print('检测每列中重复字段所占比例:')
train_offline_duplicate = \
    train_offline_dataset_feature.apply(
        lambda x: (x.value_counts().sort_values(ascending=False).max() / x.value_counts().sum()), axis=0)
print(train_offline_duplicate)
print()
test_offline_duplicate = \
    test_offline_dataset_feature.apply(
        lambda x: (x.value_counts().sort_values(ascending=False).max() / x.value_counts().sum()), axis=0)
print(test_offline_duplicate)

# # 剔除重复值程序(如果有的话)
# def drop_duplicate(x, i):
#     if ((x.value_counts().sort_values(ascending=False).max() / x.value_counts().sum()) > 0.95):
#         print('去除的特征为{0}，该特征内重复值占比为{1:.4f}\n'.format(i, (x.value_counts().sort_values(ascending=False).max() /
#                                                          x.value_counts().sum())))
#         del x


for i in np.array(test_offline_dataset_feature.columns):
    x = test_offline_dataset_feature[i]
    FT.drop_duplicate(x, i)

# (检测训练集中是否存在重复样本)
print('\n训练集是否存在重复观测: ', any(train_offline_dataset_feature.duplicated()))
train_offline_dataset_feature.drop_duplicates(inplace=True)
train_offline_dataset_feature.reset_index(level=None, drop=True, inplace=True)
print('训练集重复值处理完毕')
print(train_offline_dataset_feature.head(20))
print(train_offline_dataset_feature.info())

# (检测测试集中是否存在重复样本)
print('\n测试集是否存在重复观测: ', any(test_offline_dataset_feature.duplicated()))
test_offline_dataset_feature.drop_duplicates(inplace=True)
test_offline_dataset_feature.reset_index(level=None, drop=True, inplace=True)
print('测试集重复值处理完毕')
print(test_offline_dataset_feature.head(20))
print(test_offline_dataset_feature.info())

# 3、异常值检验
# (没有需要检测的异常值)


# 四、特征工程(对训练集和测试集同时操作,每个变量衍生步骤：训练集衍生-缺失值填充-对应测试集衍生-测试集缺失值填充)
print('--------------------特征工程--------------------')
print('训练集(特征集)特征衍生\n')
# 特征衍生
# 0、leakage特征衍生
# (1)每种用户的消费行为次数l1
# （每种user_id出现的频数）
train_assist_df_l1 = train_dataset_target['User_id'].value_counts()
train_assist_df_l1 = train_assist_df_l1.reset_index()  # Series重置索引，以便生成列名
train_assist_df_l1.columns = ['User_id', 'l1']
train_dataset_target = pd.merge(train_dataset_target, train_assist_df_l1, on='User_id', how='left')
del train_assist_df_l1
print('l1衍生完毕\n')

# (2)每种用户id领取不同种类优惠券的次数l2
train_assist_df_l2 = train_dataset_target.groupby(['User_id', 'Coupon_id'])['Date_received'].count()
train_assist_df_l2 = train_assist_df_l2.reset_index()
train_assist_df_l2.columns = ['User_id', 'Coupon_id', 'l2']
train_dataset_target = pd.merge(train_dataset_target, train_assist_df_l2, on=['User_id', 'Coupon_id'], how='left')
del train_assist_df_l2
print('l2衍生完毕\n')

# (3)每种用户id领取不同商家的次数l3
train_assist_df_l3 = train_dataset_target.groupby(['User_id', 'Merchant_id'])['Date_received'].count()
train_assist_df_l3 = train_assist_df_l3.reset_index()
train_assist_df_l3.columns = ['User_id', 'Merchant_id', 'l3']
train_dataset_target = pd.merge(train_dataset_target, train_assist_df_l3, on=['User_id', 'Merchant_id'], how='left')
del train_assist_df_l3
print('l3衍生完毕\n')

# (4)每种商品被消费行为次数l4
# （每种Merchant_id出现的频数）
train_assist_df_l4 = train_dataset_target['Merchant_id'].value_counts()
train_assist_df_l4 = train_assist_df_l4.reset_index()  # Series重置索引，以便生成列名
train_assist_df_l4.columns = ['Merchant_id', 'l4']
train_dataset_target = pd.merge(train_dataset_target, train_assist_df_l4, on='Merchant_id', how='left')
del train_assist_df_l4
print('l4衍生完毕\n')

# (5)每种用户id在此次之前领取的优惠券的次数l5
train_assist_df_l5 = train_dataset_target.groupby(['User_id'])['Date_received'].value_counts()
train_assist_df_l5 = train_assist_df_l5.reset_index(level=1, name='count')  # 可加level项删除某级索引
train_assist_df_l5.columns = ['Date_received', 'count']
train_assist_df_l5 = train_assist_df_l5.reset_index()  # 重置底层索引
train_assist_df_l5['cumsum_count'] = train_assist_df_l5.groupby(['User_id']).cumsum()  # 显示累加过程函数
train_assist_df_l5['l5'] = train_assist_df_l5['cumsum_count'] - train_assist_df_l5['count']
print('l5衍生完毕\n')

# (6)每种用户id在此次之后领取的优惠券的次数l6
train_assist_df_l6 = train_assist_df_l5.groupby(['User_id'])['count'].sum()
train_assist_df_l6 = train_assist_df_l6.reset_index()
train_assist_df_l6.columns = ['User_id', 'sum_count']
train_assist_df_l5 = pd.merge(train_assist_df_l5, train_assist_df_l6, on='User_id', how='left')
train_assist_df_l5['l6'] = train_assist_df_l5['sum_count'] - train_assist_df_l5['cumsum_count']
train_dataset_target = pd.merge(train_dataset_target, train_assist_df_l5, on=['User_id', 'Date_received'], how='left')
del train_assist_df_l5, train_assist_df_l6
train_dataset_target.drop(['count', 'sum_count', 'cumsum_count'], axis=1, inplace=True)
print('l6衍生完毕\n')

# (7)每种用户id在此次之前领取的此种优惠券的次数l7
train_assist_df_l7 = train_dataset_target.groupby(['User_id', 'Coupon_id'])['Date_received'].value_counts()
train_assist_df_l7 = train_assist_df_l7.reset_index(level=2, name='count_l8')  # 可加level项删除某级索引
train_assist_df_l7 = train_assist_df_l7.reset_index(level=1)  # 删除第二层索引
train_assist_df_l7 = train_assist_df_l7.reset_index()  # 删除底层索引
train_assist_df_l7.columns = ['User_id', 'Coupon_id', 'Date_received', 'count_l8']
train_assist_df_l7['cumsum_count_l8'] = train_assist_df_l7.groupby(['User_id', 'Coupon_id']).cumsum()  # 显示累加过程函数
train_assist_df_l7['l7'] = train_assist_df_l7['cumsum_count_l8'] - train_assist_df_l7['count_l8']
print('l7衍生完毕\n')

# (8)每种用户id在此次之后领取的此种优惠券的次数l8
train_assist_df_l8 = train_assist_df_l7.groupby(['User_id', 'Coupon_id'])['count_l8'].sum()
train_assist_df_l8 = train_assist_df_l8.reset_index()
train_assist_df_l8.columns = ['User_id', 'Coupon_id', 'sum_count_l8']
train_assist_df_l7 = pd.merge(train_assist_df_l7, train_assist_df_l8, on=['User_id', 'Coupon_id'], how='left')
train_assist_df_l7['l8'] = train_assist_df_l7['sum_count_l8'] - train_assist_df_l7['cumsum_count_l8']
train_dataset_target = pd.merge(train_dataset_target, train_assist_df_l7, on=['User_id', 'Coupon_id', 'Date_received'],
                                how='left')
del train_assist_df_l7, train_assist_df_l8
train_dataset_target.drop(['count_l8', 'sum_count_l8', 'cumsum_count_l8'], axis=1, inplace=True)
print('l8衍生完毕\n')

# (9)每种用户id在当天领取的优惠券数目l9
train_assist_df_l9 = train_dataset_target.groupby(['User_id', 'Date_received'])['Date_received'].count()
train_assist_df_l9 = train_assist_df_l9.reset_index(level=1, name='l9')  # 可加level项删除某级索引
train_assist_df_l9 = train_assist_df_l9.reset_index()  # 重置底层索引
train_dataset_target = pd.merge(train_dataset_target, train_assist_df_l9, on=['User_id', 'Date_received'], how='left')
print('l9衍生完毕\n')

# (10)每种商品id被领取的特定优惠券数目l10
train_assist_df_l10 = train_dataset_target.groupby(['Merchant_id', 'Coupon_id'])['Date_received'].count()
train_assist_df_l10 = train_assist_df_l10.reset_index(level=1, name='l10')  # 可加level项删除某级索引
train_assist_df_l10 = train_assist_df_l10.reset_index()  # 重置底层索引
train_dataset_target = pd.merge(train_dataset_target, train_assist_df_l10, on=['Merchant_id', 'Coupon_id'], how='left')
print('l10衍生完毕\n')

# (11)每种商品id被多少不同用户领取的数目l11
train_assist_df_l11 = train_dataset_target.groupby(['Merchant_id', 'User_id'])['Date_received'].count()
train_assist_df_l11 = train_assist_df_l11.reset_index(level=1, name='l11')  # 可加level项删除某级索引
train_assist_df_l11 = train_assist_df_l11.reset_index()  # 重置底层索引
train_dataset_target = pd.merge(train_dataset_target, train_assist_df_l11, on=['Merchant_id', 'User_id'], how='left')
print('l11衍生完毕\n')

# (12)某种优惠券是否是当月第一张l12
train_assist_df_l12 = train_dataset_target.groupby(['Coupon_id'])['Date_received'].value_counts()
train_assist_df_l12 = train_assist_df_l12.reset_index(level=1, name='count')  # 可加level项删除某级索引
train_assist_df_l12 = train_assist_df_l12.reset_index()  # 重置底层索引
train_assist_df_l12['cumsum_count'] = train_assist_df_l12.groupby(['Coupon_id']).cumsum()  # 显示累加过程函数
train_assist_df_l12['before'] = train_assist_df_l12['cumsum_count'] - train_assist_df_l12['count']
train_assist_df_l12.loc[train_assist_df_l12['before'] == 0, 'l12'] = 1
train_assist_df_l12['l12'].fillna(0, inplace=True)
print('l12衍生完毕\n')

# (13)某种优惠券是否是当月最后一张l13
train_assist_df_l13 = train_assist_df_l12.groupby(['Coupon_id'])['count'].sum()
train_assist_df_l13 = train_assist_df_l13.reset_index()
train_assist_df_l13.columns = ['Coupon_id', 'sum_count']
train_assist_df_l12 = pd.merge(train_assist_df_l12, train_assist_df_l13, on='Coupon_id', how='left')
train_assist_df_l12['behind'] = train_assist_df_l12['sum_count'] - train_assist_df_l12['cumsum_count']
train_assist_df_l12.loc[train_assist_df_l12['behind'] == 0, 'l13'] = 1
train_assist_df_l12['l13'].fillna(0, inplace=True)
train_dataset_target = pd.merge(train_dataset_target, train_assist_df_l12, on=['Coupon_id', 'Date_received'],
                                how='left')
del train_assist_df_l12, train_assist_df_l13
train_dataset_target.drop(['count', 'sum_count', 'cumsum_count', 'before', 'behind'], axis=1, inplace=True)
print('l13衍生完毕\n')

# (14)每种优惠券被领取的行为次数l14
train_assist_df_l14 = train_dataset_target['Coupon_id'].value_counts()
train_assist_df_l14 = train_assist_df_l14.reset_index()  # Series重置索引，以便生成列名
train_assist_df_l14.columns = ['Coupon_id', 'l14']
print(train_assist_df_l14.head(20))
train_dataset_target = pd.merge(train_dataset_target, train_assist_df_l14, on='Coupon_id', how='left')
del train_assist_df_l14
print('l14衍生完毕\n')

# (15)每种折扣比例下优惠券被领取的行为次数l15
train_assist_df_l15 = train_dataset_target['Discount_rate'].value_counts()
train_assist_df_l15 = train_assist_df_l15.reset_index()  # Series重置索引，以便生成列名
train_assist_df_l15.columns = ['Discount_rate', 'l15']
print(train_assist_df_l15.head(20))
train_dataset_target = pd.merge(train_dataset_target, train_assist_df_l15, on='Discount_rate', how='left')
del train_assist_df_l15
print('l15衍生完毕\n')

print('训练集(目标集)leakage特征衍生完毕\n')

# ------------------------------------------------
# 1、用户id角度：
# 线下:
# (1)线下领取优惠券但是没有使用的次数u1
# (2)线下普通消费次数u2
# (3)线下使用优惠券但消费在15天之后的次数u3
# (4)线下使用优惠券且15天内消费次数u4
# （首先生成消费行为变量（一个变量，3个取值1,2,3），后根据客户id分组，对每种类别进行value_counts，分组时可以单独创建df，后进行连接，u2-u4变量同理）
# (线下训练集)
train_assist_df_u1_u4 = pd.crosstab(index=train_offline_dataset_feature['User_id'],
                                    columns=train_offline_dataset_feature['off_line_action'])
train_offline_dataset_feature = pd.merge(train_offline_dataset_feature, train_assist_df_u1_u4, on='User_id', how='left')
train_offline_dataset_feature.rename(columns={1.0: 'u1', 2.0: 'u2', 3.0: 'u3', 4.0: 'u4'}, inplace=True)
del train_assist_df_u1_u4  # (删除中间变量，清理内存)
del train_offline_dataset_feature['off_line_action']
print('\nu1~u4衍生完毕\n')

# (5)每个顾客id线下领取消费券的次数u5
# （根据优惠券领取日期=null判别后求和）或（u1+u3+u4）
train_offline_dataset_feature['u5'] = train_offline_dataset_feature['u1'] + train_offline_dataset_feature['u3'] + \
                                      train_offline_dataset_feature['u4']
print('u5衍生完毕\n')

# (6)线下平均正常消费间隔u6
# （首先对客户id分组，生成每个消费日期是一年中的第几天dt.dayofyear，后最大-最小取平均）
# (线下训练集)
# 生成辅助特征'每个客户线下购买行为次数u61'
train_assist_df_u6 = train_offline_dataset_feature.groupby('User_id')['Date'].count()
train_assist_df_u6 = train_assist_df_u6.reset_index()  # 重置索引
train_assist_df_u6.rename(columns={'Date': 'u61'}, inplace=True)  # 改列名
train_offline_dataset_feature = pd.merge(train_offline_dataset_feature, train_assist_df_u6, on='User_id',
                                         how='outer')  # 连接
del train_assist_df_u6
# 生成辅助特征'每个消费日期是一年中的第几天u62'
train_offline_dataset_feature['u62'] = train_offline_dataset_feature['Date'].dt.dayofyear
# 生成辅助特征，每个客户消费日期的最大最小值以及（最大值－最小值）u63
train_assist_df_u6_max = train_offline_dataset_feature.groupby('User_id')['u62'].max()
train_assist_df_u6_min = train_offline_dataset_feature.groupby('User_id')['u62'].min()
train_assist_df_u63 = train_assist_df_u6_max - train_assist_df_u6_min
train_assist_df_u63 = train_assist_df_u63.reset_index()  # 重置索引
train_assist_df_u63.rename(columns={'u62': 'u63'}, inplace=True)  # 改列名
train_offline_dataset_feature = pd.merge(train_offline_dataset_feature, train_assist_df_u63, on='User_id',
                                         how='outer')  # 连接
train_offline_dataset_feature['u6'] = round(train_offline_dataset_feature['u63'] / train_offline_dataset_feature['u61'])
# 对u63特征进行调整，将u61=1的在u6中调整为120天(否则最大最小值为同一个值，相减为0)
train_offline_dataset_feature.loc[train_offline_dataset_feature['u61'] == 1, 'u6'] = 120  # 只购买过一次的客户的平均消费间隔设置为观察期6个月
train_offline_dataset_feature['u6'].fillna(999, inplace=True)  # 只没有购买记录客户的平均消费间隔设置为999天
del train_assist_df_u6_max, train_assist_df_u6_min, train_assist_df_u63
train_offline_dataset_feature.drop(['u62', 'u63'], axis=1, inplace=True)  # 删除新增辅助特征，清理内存
print('u6衍生完毕\n')

# (7)是否有线下消费u7
# （通过线下消费次数u61是否为0来判别）
# (线下训练集)
train_offline_dataset_feature['u7'] = np.where(train_offline_dataset_feature['u61'] == 0, 0, 1)
del train_offline_dataset_feature['u61']
print('u7衍生完毕\n')

# (8)是否有线下优惠券消费u8
# （通过线下优惠券消费次数u3+u4是否为0来判别）
# (线下训练集)
train_offline_dataset_feature['u81'] = train_offline_dataset_feature['u3'] + train_offline_dataset_feature['u4']
train_offline_dataset_feature['u8'] = np.where(train_offline_dataset_feature['u81'] == 0, 0, 1)
del train_offline_dataset_feature['u81']
print('u8衍生完毕\n')

# (9)(u3+u4)/(u2+u3+u4)用户使用优惠券的消费占比u9
# (线下训练集)
train_offline_dataset_feature['u9'] = \
    (train_offline_dataset_feature['u3'] + train_offline_dataset_feature['u4']) / \
    (train_offline_dataset_feature['u2'] + train_offline_dataset_feature['u3'] + train_offline_dataset_feature['u4'])
train_offline_dataset_feature['u9'].fillna(0,
                                           inplace=True)  # 没有购买记录客户(u2+u3+u4=0)的用户使用优惠券的消费占比设置为0,没有使用优惠券消费但有消费记录的用户(u2+u3=0)的用户使用优惠券的消费占比也为0
print('u9衍生完毕\n')

# (10)领取优惠券到使用优惠券时间间隔小于15天的次数u10
# （生成时间差变量，将小于15天记为1，大于15天和NAN记为0，对客户id分组，对时间差变量进行求和）
# (线下训练集)
train_offline_dataset_feature['u10_1'] = np.where(train_offline_dataset_feature['diff_day'] < 15, 1, 0)
train_assist_df_u10 = train_offline_dataset_feature.groupby('User_id')['u10_1'].sum()
train_assist_df_u10 = train_assist_df_u10.reset_index()
train_assist_df_u10.rename(columns={'u10_1': 'u10'}, inplace=True)
train_offline_dataset_feature = pd.merge(train_offline_dataset_feature, train_assist_df_u10, on='User_id', how='left')
train_offline_dataset_feature.drop(['u10_1', 'diff_day'], axis=1, inplace=True)
print('u10衍生完毕\n')

# (11)用户线下消费行为次数u11
# （每种user_id出现的频数）
# (线下训练集)
train_assist_df_u11 = train_offline_dataset_feature['User_id'].value_counts()
train_assist_df_u11 = train_assist_df_u11.reset_index()  # Series重置索引，以便生成列名
train_assist_df_u11.columns = ['User_id', 'u11']
train_offline_dataset_feature = pd.merge(train_offline_dataset_feature, train_assist_df_u11, on='User_id', how='left')
del train_assist_df_u11
print('u11衍生完毕\n')

# (12)生成4个月的消费周期变量-消费次数u12~u15(1月，2月，3月，4月)
# （首先提取每个客户消费时间的月份dt.month，条件筛选）
# (线下训练集)
train_offline_dataset_feature['u12_u15'] = train_offline_dataset_feature['Date'].dt.month
train_assist_df_u12_u15 = pd.crosstab(index=train_offline_dataset_feature['User_id'],
                                      columns=train_offline_dataset_feature['u12_u15'])
train_offline_dataset_feature = pd.merge(train_offline_dataset_feature, train_assist_df_u12_u15, on='User_id',
                                         how='left')
train_offline_dataset_feature.rename(columns={1.0: 'u12', 2.0: 'u13', 3.0: 'u14', 4.0: 'u15'}, inplace=True)
del train_offline_dataset_feature['u12_u15']
# 缺失值填0
train_offline_dataset_feature['u12'].fillna(0, inplace=True)
train_offline_dataset_feature['u13'].fillna(0, inplace=True)
train_offline_dataset_feature['u14'].fillna(0, inplace=True)
train_offline_dataset_feature['u15'].fillna(0, inplace=True)
print('u12~u15衍生完毕\n')

# (16)生成4个月的消费周期变量-是否消费u16~u19(1月，2月，3月，4月)
# (线下训练集)
train_offline_dataset_feature['u16'] = np.where(train_offline_dataset_feature['u12'] >= 1, 1, 0)
train_offline_dataset_feature['u17'] = np.where(train_offline_dataset_feature['u13'] >= 1, 1, 0)
train_offline_dataset_feature['u18'] = np.where(train_offline_dataset_feature['u14'] >= 1, 1, 0)
train_offline_dataset_feature['u19'] = np.where(train_offline_dataset_feature['u15'] >= 1, 1, 0)
print('u16~u19衍生完毕\n')

# (20)u10/u5用户领取优惠券到使用优惠券时间间隔小于15天的次数与线下领取优惠券的次数的比值u20
# (线下训练集)
train_offline_dataset_feature['u20'] = train_offline_dataset_feature['u10'] / train_offline_dataset_feature['u5']
# 没有线下领取过优惠券的客户缺失值填0
train_offline_dataset_feature['u20'].fillna(0, inplace=True)
print('u20衍生完毕\n')

print('训练集(特征集)线下基于用户id的特征衍生完毕\n')

# 线上:
# (1)线上用户Action_0统计uo1~uo3
# 线上用户点击次数uo1
train_assist_df_uo1_uo3 = train_online_dataset_feature.groupby('User_id')[
    'Action'].value_counts()  # 按照客户id分组，对Action变量的类别进行统计
train_assist_df_uo1_1 = train_assist_df_uo1_uo3.loc[(slice(None), [0])]  # 多重索引条件筛选，取出所有二级索引即Action为0的User_id
train_assist_df_uo1_1.index = train_assist_df_uo1_1.index.droplevel('Action')  # 去掉第二层索引,生成Series
train_assist_df_uo1_2 = train_assist_df_uo1_1.reset_index()
train_assist_df_uo1_2.rename(columns={'Action': 'uo1'}, inplace=True)
train_online_dataset_feature = pd.merge(train_online_dataset_feature, train_assist_df_uo1_2, on='User_id',
                                        how='left')  # 连接线上特征每个用户id对应Action的次数
del train_assist_df_uo1_1, train_assist_df_uo1_2

# 线上用户购买次数uo2
train_assist_df_uo2_1 = train_assist_df_uo1_uo3.loc[(slice(None), [1])]  # 取出所有二级索引即Action为1的User_id
train_assist_df_uo2_1.index = train_assist_df_uo2_1.index.droplevel('Action')
train_assist_df_uo2_2 = train_assist_df_uo2_1.reset_index()
train_assist_df_uo2_2.rename(columns={'Action': 'uo2'}, inplace=True)
train_online_dataset_feature = pd.merge(train_online_dataset_feature, train_assist_df_uo2_2, on='User_id', how='left')
del train_assist_df_uo2_1, train_assist_df_uo2_2

# 线上用户领取优惠券次数uo3
train_assist_df_uo3_1 = train_assist_df_uo1_uo3.loc[(slice(None), [2])]
train_assist_df_uo3_1.index = train_assist_df_uo3_1.index.droplevel('Action')
train_assist_df_uo3_2 = train_assist_df_uo3_1.reset_index()
train_assist_df_uo3_2.rename(columns={'Action': 'uo3'}, inplace=True)
train_online_dataset_feature = pd.merge(train_online_dataset_feature, train_assist_df_uo3_2, on='User_id', how='left')
del train_assist_df_uo3_1, train_assist_df_uo3_2
del train_assist_df_uo1_uo3
print('\nuo1~uo3特征衍生完毕\n')

# (4)线上行为次数uo4
# （每种user_id线上出现的频数）
train_assist_df_uo4 = train_online_dataset_feature['User_id'].value_counts()
train_assist_df_uo4 = train_assist_df_uo4.reset_index()  # Series重置索引，以便生成列名
train_assist_df_uo4.columns = ['User_id', 'uo4']
train_online_dataset_feature = pd.merge(train_online_dataset_feature, train_assist_df_uo4, on='User_id', how='left')
del train_assist_df_uo4
print('uo4特征衍生完毕\n')

# 将线上特征拼接到线下训练集
train_offline_dataset_assist_df = train_online_dataset_feature[['User_id', 'uo1', 'uo2', 'uo3', 'uo4']]
train_offline_dataset_assist_df.drop_duplicates(inplace=True)  # 生成不重复客户数据集，防止踩坑merge
train_offline_dataset_feature = pd.merge(train_offline_dataset_feature, train_offline_dataset_assist_df, on='User_id',
                                         how='left')
del train_offline_dataset_assist_df

# 对连接到线下后的线上特征进行缺失值填充
train_offline_dataset_feature['uo1'].fillna(0, inplace=True)
train_offline_dataset_feature['uo2'].fillna(0, inplace=True)
train_offline_dataset_feature['uo3'].fillna(0, inplace=True)
train_offline_dataset_feature['uo4'].fillna(0, inplace=True)

# (5)是否有线上行为uo5，是1，否0（在train_online_dataset_feature是否出现）
# （在线上表格中是否对应，线上表格中提取不重复的客户列和全1列，生成新DateFrame，与线下数据进行拼接）
online_unique_user = train_online_dataset_feature['User_id'].unique()
ones_array = np.ones((len(online_unique_user),))  # (683053,)
train_assist_df_uo5 = pd.Series(data=ones_array, index=online_unique_user)
train_assist_df_uo5 = train_assist_df_uo5.reset_index()  # Series重置索引，以便生成列名
train_assist_df_uo5.columns = ['User_id', 'uo5']
train_offline_dataset_feature = pd.merge(train_offline_dataset_feature, train_assist_df_uo5, on='User_id', how='left')
train_offline_dataset_feature['uo5'].fillna(0, inplace=True)  # 线上表中未出现用户填0
del train_assist_df_uo5
print('uo5衍生完毕\n')
print('训练集(特征集)线上基于用户id的特征衍生完毕\n')

# 对应训练集(目标集)变量衍生
train_target_assist_df = \
    train_offline_dataset_feature[['User_id', 'u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7', 'u8'
        , 'u9', 'u10', 'u11', 'u12', 'u13', 'u14', 'u15', 'u16', 'u17', 'u18', 'u19', 'u20', 'uo1', 'uo2', 'uo3', 'uo4',
                                   'uo5']]  # (里面以客户id分组均为重复行)
train_target_assist_df.drop_duplicates(inplace=True)  # 生成不重复客户数据集，防止踩坑merge
train_dataset_target = pd.merge(train_dataset_target, train_target_assist_df, on='User_id', how='left')
del train_target_assist_df

# 对应训练集(目标集)缺失值填充
for column in ['u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7', 'u8', 'u10', 'u11', 'u12', 'u13', 'u14', 'u15', 'u16', 'u17',
               'u18', 'u19', 'uo1', 'uo2', 'uo3', 'uo4', 'uo5']:
    mean_val = train_dataset_target[column].mean()
    train_dataset_target[column].fillna(round(mean_val), inplace=True)
# u9,u20为非整数变量，单独用平均值填充
train_dataset_target['u9'].fillna(train_dataset_target['u9'].mean(), inplace=True)
train_dataset_target['u20'].fillna(train_dataset_target['u9'].mean(), inplace=True)

# ------------------------------------------------
# 2、商户角度：
# (1)每种商品线下热度m1
# （每种商品id线下出现的次数）
# (线下训练集)
train_assist_df_m1 = train_offline_dataset_feature['Merchant_id'].value_counts()  # series
train_assist_df_m1 = train_assist_df_m1.reset_index()
train_assist_df_m1.columns = ['Merchant_id', 'm1']
train_offline_dataset_feature = pd.merge(train_offline_dataset_feature, train_assist_df_m1, on='Merchant_id',
                                         how='left')  # 连接线下特征每个商户id对应出现的次数
del train_assist_df_m1
print('\nm1特征衍生完毕\n')

# (2)每种商品被购买的次数m2
# （按照商品id分类，对Date进行求和）
# (线下训练集)
train_assist_df_m2 = train_offline_dataset_feature.groupby('Merchant_id')['Date'].count()
train_assist_df_m2 = train_assist_df_m2.reset_index()
train_assist_df_m2.columns = ['Merchant_id', 'm2']
train_offline_dataset_feature = pd.merge(train_offline_dataset_feature, train_assist_df_m2, on='Merchant_id',
                                         how='left')
del train_assist_df_m2
print('m2特征衍生完毕\n')

# (3)每种商品使用优惠券购买的次数m3
# （原理同u3，把客户id换成商品id即可）
# (线下训练集)
# 先生成商品消费行为变量1-领券但未购买，2-普通消费，3-领券且购买
train_offline_dataset_feature['Date'].fillna('nan', inplace=True)
train_offline_dataset_feature['Date_received'].fillna('nan', inplace=True)
train_offline_dataset_feature.loc[
    train_offline_dataset_feature['Date'] == 'nan', 'off_line_Merchant_id_action'] = 1  # 领券但未购买
train_offline_dataset_feature.loc[
    train_offline_dataset_feature['Date_received'] == 'nan', 'off_line_Merchant_id_action'] = 2  # 普通消费
train_offline_dataset_feature['off_line_Merchant_id_action'].fillna(3, inplace=True)  # 领券且购买
train_assist_df_m3 = \
    train_offline_dataset_feature.groupby(['Merchant_id'])[
        'off_line_Merchant_id_action'].value_counts()  # 按照商户id分组，对off_line_Merchant_id_action变量的类别进行统计
train_assist_df_m31 = train_assist_df_m3.loc[(slice(None), [3])]
train_assist_df_m31.index = train_assist_df_m31.index.droplevel('off_line_Merchant_id_action')
train_assist_df_m32 = train_assist_df_m31.reset_index()
train_assist_df_m32.rename(columns={'off_line_Merchant_id_action': 'm3'}, inplace=True)
train_offline_dataset_feature = pd.merge(train_offline_dataset_feature, train_assist_df_m32, on='Merchant_id',
                                         how='left')
del train_assist_df_m31, train_assist_df_m32
# 缺失值填0
train_offline_dataset_feature['m3'].fillna(0, inplace=True)
print('m3特征衍生完毕\n')

# (4)m2/m1商品线下被购买率m4
# (线下训练集)
train_offline_dataset_feature['m4'] = train_offline_dataset_feature['m2'] / train_offline_dataset_feature['m1']
print('m4特征衍生完毕\n')

# (5)每种商品发放优惠券的数目m5
# （按照商品id分类，对优惠券id进行求和）
# (线下训练集)
train_assist_df_m4 = train_offline_dataset_feature.groupby('Merchant_id')['Coupon_id'].count()
train_assist_df_m4 = train_assist_df_m4.reset_index()
train_assist_df_m4.columns = ['Merchant_id', 'm5']
train_offline_dataset_feature = pd.merge(train_offline_dataset_feature, train_assist_df_m4, on='Merchant_id',
                                         how='left')
del train_assist_df_m4
print('m5特征衍生完毕\n')

# (6)m3/m5每种商品优惠券使用率m6
# (线下训练集)
train_offline_dataset_feature['m6'] = train_offline_dataset_feature['m3'] / train_offline_dataset_feature['m5']
# 将没有发放优惠券的商品(m5=0)缺失值填充为0
train_offline_dataset_feature['m6'].fillna(0, inplace=True)
print('m6特征衍生完毕\n')
print('训练集(特征集)线下基于商户id的特征衍生完毕\n')

# 训练集(目标集)变量衍生
train_target_assist_df = train_offline_dataset_feature[['Merchant_id', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6']]
train_target_assist_df.drop_duplicates(inplace=True)  # 生成不重复商户数据集，防止踩坑merge
train_dataset_target = pd.merge(train_dataset_target, train_target_assist_df, on='Merchant_id', how='left')
del train_target_assist_df

# 训练集(目标集)缺失值填充
for column in ['m1', 'm2', 'm3', 'm5']:
    mean_val = train_dataset_target[column].mean()
    train_dataset_target[column].fillna(round(mean_val), inplace=True)
# m4,m6为非整数变量，单独用平均值填充
train_dataset_target['m4'].fillna(train_dataset_target['m4'].mean(), inplace=True)
train_dataset_target['m6'].fillna(train_dataset_target['m6'].mean(), inplace=True)

# ------------------------------------------------
# 3、优惠券角度：
# (1)优惠券领取日期是周几c1
# （dt.dayofweek）
train_offline_dataset_feature['Date_received'] = pd.to_datetime(train_offline_dataset_feature['Date_received'])
train_offline_dataset_feature['Date'] = pd.to_datetime(train_offline_dataset_feature['Date'])
train_dataset_target['Date_received'] = pd.to_datetime(train_dataset_target['Date_received'])
# (训练集(特征集))
train_offline_dataset_feature['c1'] = train_offline_dataset_feature['Date_received'].dt.dayofweek
# 缺失值用7填充(周一到周日为0-6)
train_offline_dataset_feature['c1'].fillna(7, inplace=True)
# (训练集(目标集))
train_dataset_target['c1'] = train_dataset_target['Date_received'].dt.dayofweek
print('\nc1特征衍生完毕\n')

# (2)优惠券领取日期是否为工作日c2
# (dt.dayofweek中5,6为周末)
# (训练集(特征集))
train_offline_dataset_feature.loc[train_offline_dataset_feature['c1'] == 5, 'c2'] = 1
train_offline_dataset_feature.loc[train_offline_dataset_feature['c1'] == 6, 'c2'] = 1
# 除5,6外其他为非工作日，填0
train_offline_dataset_feature['c2'].fillna(0, inplace=True)
# (训练集(目标集))
train_dataset_target.loc[train_dataset_target['c1'] == 5, 'c2'] = 1
train_dataset_target.loc[train_dataset_target['c1'] == 6, 'c2'] = 1
# 除5,6外其他为非工作日，填0
train_dataset_target['c2'].fillna(0, inplace=True)
print('c2特征衍生完毕\n')

# (3)每种优惠券共发行多少张c3
# （对优惠券id进行分组后计数）
# (训练集(特征集))
train_assist_df_c3 = train_offline_dataset_feature['Coupon_id'].value_counts()  # series
train_assist_df_c3 = train_assist_df_c3.reset_index()
train_assist_df_c3.columns = ['Coupon_id', 'c3']
train_offline_dataset_feature = pd.merge(train_offline_dataset_feature, train_assist_df_c3, on='Coupon_id',
                                         how='left')  # 连接线下特征每个商户id对应出现的次数
# 缺失值填0，说明训练集中未发行此种优惠券
train_offline_dataset_feature['c3'].fillna(0, inplace=True)
del train_assist_df_c3
# (训练集(目标集))
test_assist_df_c3 = train_offline_dataset_feature[['Coupon_id', 'c3']]
test_assist_df_c3.drop_duplicates(inplace=True)  # 生成不重复商户数据集，防止踩坑merge
train_dataset_target = pd.merge(train_dataset_target, test_assist_df_c3, on='Coupon_id', how='left')
# 测试集缺失值填0，说明训练集中未发行此种优惠券
train_dataset_target['c3'].fillna(0, inplace=True)
del test_assist_df_c3
print('c3特征衍生完毕\n')

# (4)优惠券折扣力度c4
# （30:5--(30-5)/30,0.7--0.7）
# (训练集(特征集))
train_offline_dataset_feature['c4'] = train_offline_dataset_feature['Discount_rate']

train_offline_dataset_feature['c4'] = train_offline_dataset_feature['c4'].apply(FT.Disc_Cal)
train_offline_dataset_feature['c4'].fillna(1, inplace=True)
# (训练集(目标集))
train_dataset_target['c4'] = train_dataset_target['Discount_rate']
train_dataset_target['c4'] = train_dataset_target['c4'].apply(FT.Disc_Cal)
print('c4特征衍生完毕\n')

# (5)优惠券可以使用的最低消费额度c5
# （30:5--30,0.7--0）
# (训练集(特征集))
train_offline_dataset_feature['c5'] = train_offline_dataset_feature['Discount_rate']

train_offline_dataset_feature['c5'] = train_offline_dataset_feature['c5'].apply(FT.Lowest_Disc_Cal)
# 填1而不填0的目的：与0.x的消费券区分开
train_offline_dataset_feature['c5'].fillna(1, inplace=True)
# (训练集(目标集))
train_dataset_target['c5'] = train_dataset_target['Discount_rate']
train_dataset_target['c5'] = train_dataset_target['c5'].apply(FT.Lowest_Disc_Cal)
print('c4特征衍生完毕\n')
print('线下训练集(优惠券角度)特征衍生完毕\n')

# ------------------------------------------------
# 4、用户和商户组合特征：（看能否与测试集对应上进行取舍）
# (1)各用户在各商品上的领券但未使用次数um1
# （对用户id和商品id同时进行分组，对线下行为1变量进行求和）
train_assist_df_um1_um3 = \
    train_offline_dataset_feature.groupby(['User_id', 'Merchant_id'])[
        'off_line_Merchant_id_action'].value_counts()  # 三重索引
train_assist_df_um11 = train_assist_df_um1_um3.loc[(slice(None), slice(None), [1])]
train_assist_df_um11.index = train_assist_df_um11.index.droplevel('off_line_Merchant_id_action')
train_assist_df_um12 = train_assist_df_um11.reset_index()
train_assist_df_um12.rename(columns={'off_line_Merchant_id_action': 'um1'}, inplace=True)
train_offline_dataset_feature = \
    pd.merge(train_offline_dataset_feature, train_assist_df_um12, on=['User_id', 'Merchant_id'], how='left')
del train_assist_df_um11, train_assist_df_um12
# 缺失值用0次填充
train_offline_dataset_feature['um1'].fillna(0, inplace=True)
print('\num1特征衍生完毕\n')

# (2)各用户在各商品上的普通消费次数um2
train_assist_df_um21 = train_assist_df_um1_um3.loc[(slice(None), slice(None), [2])]
train_assist_df_um21.index = train_assist_df_um21.index.droplevel('off_line_Merchant_id_action')
train_assist_df_um22 = train_assist_df_um21.reset_index()
train_assist_df_um22.rename(columns={'off_line_Merchant_id_action': 'um2'}, inplace=True)
train_offline_dataset_feature = pd.merge(train_offline_dataset_feature, train_assist_df_um22,
                                         on=['User_id', 'Merchant_id'], how='left')
del train_assist_df_um21, train_assist_df_um22
# 缺失值用0次填充
train_offline_dataset_feature['um2'].fillna(0, inplace=True)
print('um2特征衍生完毕\n')

# (3)各用户在各商品上的领取过优惠券且消费次数um3
train_assist_df_um31 = train_assist_df_um1_um3.loc[(slice(None), slice(None), [3])]
train_assist_df_um31.index = train_assist_df_um31.index.droplevel('off_line_Merchant_id_action')
train_assist_df_um32 = train_assist_df_um31.reset_index()
train_assist_df_um32.rename(columns={'off_line_Merchant_id_action': 'um3'}, inplace=True)
train_offline_dataset_feature = pd.merge(train_offline_dataset_feature, train_assist_df_um32,
                                         on=['User_id', 'Merchant_id'], how='left')
del train_assist_df_um31, train_assist_df_um32
# 缺失值用0次填充
train_offline_dataset_feature['um3'].fillna(0, inplace=True)
print('um3特征衍生完毕\n')

# (4)um1+um3各用户在各商品上的领取过优惠券的次数um4
train_offline_dataset_feature['um4'] = train_offline_dataset_feature['um1'] + train_offline_dataset_feature['um3']
print('um4特征衍生完毕\n')

# (5)um2+um3各用户在各商品上的消费次数um5
train_offline_dataset_feature['um5'] = train_offline_dataset_feature['um2'] + train_offline_dataset_feature['um3']
print('um5特征衍生完毕\n')
print('线下训练集(交互角度)特征衍生完毕\n')
del train_offline_dataset_feature['off_line_Merchant_id_action']

# 训练集(目标集)变量衍生
train_target_assist_df = train_offline_dataset_feature.loc[:,
                         ['User_id', 'Merchant_id', 'um1', 'um2', 'um3', 'um4', 'um5']]
train_target_assist_df.drop_duplicates(inplace=True)
train_dataset_target = pd.merge(train_dataset_target, train_target_assist_df, on=['User_id', 'Merchant_id'], how='left')
del train_target_assist_df

# 训练集(目标集)变量衍生
# 在按照客户id和商户id对应到测试集后测试集没有的信息填0次
for column in ['um1', 'um2', 'um3', 'um4', 'um5']:
    train_dataset_target[column].fillna(0, inplace=True)
print('\n训练集特征衍生结束\n')

# 生成新的训练集
train_dataset_target.to_csv(r'E:\PycharmProjects\sklearn\Data\o2o_coupon\origin_train_dataset.csv', index=False)
print('\n训练集保存完毕\n')

print("DONE")

'''print('\n训练集信息')
print(train_dataset_target.head(20))
print(train_dataset_target.info())'''
