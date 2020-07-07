#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
删除无关特征,目标变量放在第一列
特征编码：连续特征进行归一化、分类特征进行热编码
卡方检验选变量，生成相关系数矩阵，去除相关系数>0.8的中卡方值较小的变量
多重共线性检验、平衡样本集(smote采样 OR NOT)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from statsmodels.stats.outliers_influence import variance_inflation_factor
import Fuction_Total as FT

# 消除版本引起的参数设置警告
import warnings

warnings.filterwarnings("ignore")

# 数据导入+查看
# os.chdir(r'E:\Data\o2o优惠券使用预测')  # 设置工作路径

origin_train_dataset = pd.read_csv(r'E:\PycharmProjects\sklearn\Data\o2o_coupon\origin_train_dataset.csv',
                                   encoding='utf-8')
# 线下优惠券领取日期和购买日期转换为时间格式
origin_train_dataset['Date_received'] = pd.to_datetime(origin_train_dataset['Date_received'])
origin_train_dataset['Date'] = pd.to_datetime(origin_train_dataset['Date'])

origin_test_dataset = pd.read_csv(r'E:\PycharmProjects\sklearn\Data\o2o_coupon\origin_test_dataset.csv',
                                  encoding='utf-8')
# 测试集优惠券领取日期和购买日期转换为时间格式
origin_test_dataset['Date_received'] = pd.to_datetime(origin_test_dataset['Date_received'])

print('--------------------特征工程--------------------')
# 1、在训练集和测试集中删除无关特征,目标变量放在第一列
origin_train_dataset.drop(['Date_received', 'Date'], axis=1, inplace=True)
origin_test_dataset.drop(['Date_received'], axis=1, inplace=True)
# 将目标列放在第一列
target = origin_train_dataset['target']
origin_train_dataset.drop(labels=['target'], axis=1, inplace=True)
origin_train_dataset.insert(0, 'target', target)
# 处理新生成的训练集和数据集在distance/优惠券id上的缺失值
# 训练集Distance用平均数(取整)填充
origin_train_dataset['Distance'].fillna(round(origin_train_dataset['Distance'].mean()), inplace=True)
# 测试集Distance用平均数(取整)填充
origin_test_dataset['Distance'].fillna(round(origin_test_dataset['Distance'].mean()), inplace=True)
# 训练集Coupon_id用000填充
origin_train_dataset['Coupon_id'].fillna(000, inplace=True)
# 测试集Coupon_id用000填充
origin_test_dataset['Coupon_id'].fillna(000, inplace=True)

# 去掉对应编码不同的特征'User_id','Merchant_id','Coupon_id','Discount_rate'
# 训练集
origin_train_dataset.drop(['User_id', 'Merchant_id', 'Coupon_id', 'Discount_rate'], axis=1, inplace=True)
# 测试集
origin_test_dataset.drop(['User_id', 'Merchant_id', 'Coupon_id', 'Discount_rate'], axis=1, inplace=True)

# print(len(origin_train_dataset['Discount_rate'].value_counts().values)) #41种
# print(len(origin_test_dataset['Discount_rate'].value_counts().values)) #42种

# 2、特征编码
con_feature = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u9', 'u10', 'u11', 'u12', 'u13', 'u14', 'u15', 'u20', 'uo1', 'uo2',
               'uo3', 'uo4',
               'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'c3', 'c4', 'um1', 'um2', 'um3', 'um4', 'um5', 'c5', 'l1', 'l2',
               'l3', 'l4', 'l5',
               'l6', 'l7', 'l8', 'l9', 'l10', 'l11', 'l14', 'l15']
# 这里去掉Discount_rate特征，obj类型无法进行卡方检验
sep_feature = ['Distance', 'u7', 'u8', 'u16', 'u17', 'u18', 'u19', 'c1', 'c2', 'uo5', 'l12', 'l13']
# 连续特征进行归一化
mms = MinMaxScaler()  # 实例化归一化函数
for c1 in con_feature:
    origin_train_dataset[c1] = mms.fit_transform(origin_train_dataset[[c1]])
    origin_test_dataset[c1] = mms.fit_transform(origin_test_dataset[[c1]])
print('特征归一化结束\n')

print('训练集信息：')
print(origin_train_dataset.loc[:, sep_feature].head(20))
print('\n测试集集信息：')
print(origin_test_dataset.loc[:, sep_feature].head(20))
# 分类特征进行热编码(该步编码为为了数值化特征，使其可以输入到卡方检验中)
le = LabelEncoder()  # 实例化
for c2 in sep_feature:
    origin_train_dataset[c2] = le.fit_transform(origin_train_dataset[c2].astype(str))
    origin_test_dataset[c2] = le.fit_transform(origin_test_dataset[c2].astype(str))
print('特征编码结束\n')
print('训练集信息：')
print(origin_train_dataset.loc[:, sep_feature].head(20))
print('\n测试集集信息：')
print(origin_test_dataset.loc[:, sep_feature].head(20))

# 3、卡方检验
pd.set_option('float_format', lambda x: '%.3f' % x)  # 取消科学计数法
# 利用卡方检验，选择k个最佳特征
feature = con_feature + sep_feature
feature_k = SelectKBest(chi2, k=len(feature))  # 实例化卡方检验函数
fea_new = feature_k.fit_transform(origin_train_dataset.loc[:, feature], origin_train_dataset.loc[:, 'target'])
data = {'feature': feature, 'fea_new_scores': feature_k.scores_, 'fea_new_pvalues': feature_k.pvalues_}
fea_importance = pd.DataFrame(data)
fea_importance = fea_importance.sort_values(by='fea_new_scores', ascending=False)
print(fea_importance)  # 除文本特征外各特征的重要度以及对应得分
fea_selected_chi = list(fea_importance[fea_importance['fea_new_pvalues'] < 0.005]['feature'])  # 筛选p值<0.05的特征
print('\n卡方检验特征筛选完毕，筛选出来的特征个数为：{}个'.format(len(fea_selected_chi)))
fea_selected_chi_train_df = origin_train_dataset.loc[:, fea_selected_chi]
fea_selected_chi_test_df = origin_test_dataset.loc[:, fea_selected_chi]

# 4、生成相关系数矩阵，去除相关系数>0.8的中卡方值较小的变量(适度筛选)
del_var = ['u16', 'm1', 'u5', 'u2', 'u4', 'u7']
for i in del_var:
    fea_selected_chi.remove(i)
corr_df = fea_selected_chi_train_df.loc[:, fea_selected_chi].corr()
fig, ax = plt.subplots(figsize=(10, 10))
# robust=True自动设置颜色,annot显示网格数据,fmt小数保留位数
ax = sns.heatmap(corr_df, linewidths=0.5, vmax=1.2, vmin=-1.2, cmap='rainbow', annot=False, fmt='.2f')
plt.show()

# 5、进行多重共线性检验，去除有多重共线性的变量(um1)
# 该部分代码运行极慢，5分钟左右
fea_selected_chi.remove('um1')
VIF = pd.DataFrame()
vif_df = fea_selected_chi_train_df.loc[:, fea_selected_chi]
VIF["features"] = vif_df.columns
VIF["VIF Factor"] = [variance_inflation_factor(vif_df.values, i) for i in range(vif_df.shape[1])]
print(VIF)

# 6、平衡样本集，smote采样(生成正样本)
# smo = SMOTE(sampling_strategy={1:50000},random_state=111)
# X_smo,y_smo=smo.fit_sample(fea_selected_chi_train_df.loc[:,fea_selected_chi],origin_train_dataset.loc[:,'target'])
# final_train_dataset=pd.concat([X_smo,y_smo],axis = 1)
##生成最终测试集
# final_test_dataset=origin_test_dataset.loc[:,fea_selected_chi]
#
# print('训练集信息：')
# print(final_train_dataset.head(20))
# print(final_train_dataset.info(verbose=True,null_counts=True))
# print('\n测试集集信息：')
# print(final_test_dataset.head(20))
# print(final_test_dataset.info(verbose=True,null_counts=True))


# 不进行样本集平衡
final_test_dataset = origin_test_dataset.loc[:, fea_selected_chi]
final_train_dataset = pd.concat(
    [fea_selected_chi_train_df.loc[:, fea_selected_chi], origin_train_dataset.loc[:, 'target']], axis=1)
print(final_train_dataset['target'].value_counts())
# 生成最终的训练集，测试集
final_train_dataset.to_csv(r'E:\PycharmProjects\sklearn\Data\o2o_coupon\final_train_dataset.csv', index=False)
final_test_dataset.to_csv(r'E:\PycharmProjects\sklearn\Data\o2o_coupon\final_test_dataset.csv', index=False)

print('训练集信息：')
print(final_train_dataset.head(20))
print(final_train_dataset.info())
print('\n测试集集信息：')
print(final_test_dataset.head(20))
print(final_test_dataset.info())

print("DONE")
