#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KNN、LR、Adaboost、RF、LGB、XGB
"""

import os
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
import lightgbm
import xgboost
import Fuction_Total as FT

# 消除版本引起的参数设置警告
import warnings

warnings.filterwarnings("ignore")

# 数据导入+查看
# os.chdir(r'E:\Data\o2o优惠券使用预测')  # 设置工作路径

final_train_dataset = pd.read_csv(r'E:\PycharmProjects\sklearn\Data\o2o_coupon\final_train_dataset.csv',
                                  encoding='utf-8')
final_test_dataset = pd.read_csv(r'E:\PycharmProjects\sklearn\Data\o2o_coupon\final_test_dataset.csv', encoding='utf-8')
submit_test_dataset = pd.read_csv(r'E:\PycharmProjects\sklearn\Data\o2o_coupon\test_dataset_target.csv',
                                  encoding='utf-8')

print('训练集信息：')
print(final_train_dataset.head(20))
print(final_train_dataset.info(verbose=True, null_counts=True))
print('\n测试集集信息：')
print(final_test_dataset.head(20))
print(final_test_dataset.info(verbose=True, null_counts=True))

# print('--------------------特征工程--------------------')
##分类特征进行one-hot编码(放在特征工程最后)
##分类变量中去除了uo5未通过卡方检验
# sep_feature_selected=['Distance','u7','u8','u16','u17','u18','u19','c1','c2','User_id','Merchant_id','Coupon_id']
# enc = OneHotEncoder(sparse=False)#实例化编码函数
# for c in sep_feature_selected:
# 	final_train_dataset[c] = enc.fit_transform(final_train_dataset[[c]])
# for c in sep_feature_selected:
# 	final_test_dataset[c] = enc.fit_transform(final_test_dataset[[c]])
# print('\n特征工程结束\n')


# 七、模型搭建与调参
print('--------------------模型搭建与调参--------------------')
# 划分训练集
X_var = list(final_train_dataset.columns)
X_var.remove('target')  # 去掉'target'列
X = np.array(final_train_dataset[X_var].values)
Y = final_train_dataset['target']
validation_size = 0.20  # 训练集：测试集=8:2
seed = 100
X_train, X_validation, Y_train, Y_validation = \
    model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)  # 添加随机数种子使得每次运行的验证集和训练集是不变的
print('\n训练集划分完毕\n')

# 第一张：KNN&LR&Adaboost&RF
plt.style.use('ggplot')  # 设置绘图风格
# 生成画板以及各个子图
fig = plt.figure('ROC', figsize=(10, 12))
ax1 = plt.subplot2grid((4, 4), (0, 0), rowspan=2, colspan=2)
ax2 = plt.subplot2grid((4, 4), (0, 2), rowspan=2, colspan=2)
ax3 = plt.subplot2grid((4, 4), (2, 0), rowspan=2, colspan=2)
ax4 = plt.subplot2grid((4, 4), (2, 2), rowspan=2, colspan=2)

# 搭建模型1(K近邻)
time_1 = time.time()
KNN = KNeighborsClassifier()
KNN.fit(X_train, Y_train)  # 传入训练数据
Y_pre_knn = KNN.predict_proba(X_validation)  # Y_pre与测试集的标签相匹配而不是和整个数据集的标签相匹配
Y_pre_knn = np.array(Y_pre_knn[:, 1])  # 取第二列数据，因为第二列概率为趋于0时分类类别为0，概率趋于1时分类类别为1
fpr, tpr, thresholds = roc_curve(Y_validation, Y_pre_knn)  # 计算fpr,tpr,thresholds
auc_knn = roc_auc_score(Y_validation, Y_pre_knn)  # 计算auc
print('\nKNN模型AUC值为：{:.6f}\n'.format(auc_knn))
print('K近邻耗时：{0}s\n'.format(round(time.time() - time_1, 2)))
print()
# 画ROC曲线图
plt.sca(ax2)
plt.xlim(-0.02, 1.02)
plt.ylim(0, 1.02)
plt.plot(fpr, tpr, 'b')
plt.title('$ROC curve-KNN$')
plt.plot([0, 1], [0, 1], 'r--')
plt.fill_between(fpr, tpr, color='lightgreen', alpha=0.6)
plt.text(0.1, 0.65, 'AUC = {:.4f}'.format(auc_knn), fontsize=15)

# 搭建模型2(逻辑回归)
time_2 = time.time()
LR = LogisticRegression(solver='newton-cg', multi_class='ovr')  # 创建模型
LR.fit(X_train, Y_train)  # 传入训练数据
Y_pre_lr = LR.predict_proba(X_validation)  # Y_pre与测试集的标签相匹配而不是和整个数据集的标签相匹配
Y_pre_lr = np.array(Y_pre_lr[:, 1])  # 取第二列数据，因为第二列概率为趋于0时分类类别为0，概率趋于1时分类类别为1
fpr, tpr, thresholds = roc_curve(Y_validation, Y_pre_lr)  # 计算fpr,tpr,thresholds
auc_lr = roc_auc_score(Y_validation, Y_pre_lr)  # 计算auc
print('\nLR模型AUC值为：{:.6f}\n'.format(auc_lr))
print('逻辑回归耗时：{0}s\n'.format(round(time.time() - time_2, 2)))
print()
# 画ROC曲线图
plt.sca(ax1)
plt.xlim(-0.02, 1.02)
plt.ylim(0, 1.02)
plt.plot(fpr, tpr, 'b')
plt.title('$ROC curve-LR$')
plt.plot([0, 1], [0, 1], 'r--')
plt.fill_between(fpr, tpr, color='lightgreen', alpha=0.6)
plt.text(0.1, 0.65, 'AUC = {:.4f}'.format(auc_lr), fontsize=15)

# 搭建模型3(adaboost)
time_3 = time.time()
Ada = AdaBoostClassifier()
Ada.fit(X_train, Y_train)  # 传入训练数据
Y_pre_ada = Ada.predict_proba(X_validation)  # Y_pre与测试集的标签相匹配而不是和整个数据集的标签相匹配
Y_pre_ada = np.array(Y_pre_ada[:, 1])  # 取第二列数据，因为第二列概率为趋于0时分类类别为0，概率趋于1时分类类别为1
fpr, tpr, thresholds = roc_curve(Y_validation, Y_pre_ada)  # 计算fpr,tpr,thresholds
auc_ada = roc_auc_score(Y_validation, Y_pre_ada)  # 计算auc
print('\nAdaboost模型AUC值为：{:.6f}\n'.format(auc_ada))
print('Adaboost耗时：{0}s\n'.format(round(time.time() - time_3, 2)))
print()
# 画ROC曲线图
plt.sca(ax3)
plt.xlim(-0.02, 1.02)
plt.ylim(0, 1.02)
plt.plot(fpr, tpr, 'b')
plt.title('$ROC curve-Adaboost$')
plt.plot([0, 1], [0, 1], 'r--')
plt.fill_between(fpr, tpr, color='lightgreen', alpha=0.6)
plt.text(0.1, 0.65, 'AUC = {:.4f}'.format(auc_ada), fontsize=15)

# 搭建模型4(随机森林)
# 网格调参
# time_4=time.time()
# tuned_parameters=[{'n_estimators': [20,50,100,200],'max_depth': [5,10,20], 'max_features': [5,10,15,20]}]
# scores = 'precision'
# 调用 GridSearchCV，将RandomForestClassifier(random_state=0),tuned_parameters,cv=10,还有 scoring 传递进去，
# rfc=GridSearchCV(RandomForestClassifier(random_state=0), tuned_parameters, cv=10,scoring=scores)
# 用训练集训练这个学习器 clf
# rfc.fit(X_train, Y_train)
# print("Best parameters set found on development set:")
# 再调用 rfc.best_params_ 就能直接得到最好的参数搭配结果
# print(rfc.best_params_)
# print('随机森林调参耗时：{0}'.format(time.time()-time_4))
# 测试集精度及AUC值
time_4 = time.time()
rfc = RandomForestClassifier()  # 0.885897
rfc.fit(X_train, Y_train)
Y_pre_rf = rfc.predict_proba(X_validation)
Y_pre_rf = np.array(Y_pre_rf[:, 1])
fpr, tpr, thresholds = roc_curve(Y_validation, Y_pre_rf)
auc_rf = roc_auc_score(Y_validation, Y_pre_rf)
print('RF模型的AUC值为：{:.6f}\n'.format(auc_rf))
print('随机森林耗时：{0}s\n'.format(round(time.time() - time_4, 2)))
print()
# 画ROC曲线图
plt.sca(ax4)
plt.xlim(-0.02, 1.02)
plt.ylim(0, 1.02)
plt.plot(fpr, tpr, 'b')
plt.title('$ROC curve-RF$')
plt.plot([0, 1], [0, 1], 'r--')
plt.fill_between(fpr, tpr, color='lightgreen', alpha=0.6)
plt.text(0.1, 0.65, 'AUC = {:.4f}'.format(auc_rf), fontsize=15)
plt.show()

# 第二张：LGB&XGB
fig2 = plt.figure('ROC', figsize=(8, 10))
ax1 = plt.subplot2grid((1, 2), (0, 0))
ax2 = plt.subplot2grid((1, 2), (0, 1))

# 搭建模型5(LightGBM)
time_5 = time.time()
modellgb = lightgbm.LGBMRegressor(learning_rate=0.05, max_depth=10, feature_fraction=0.9, bagging_fraction=1,
                                  bagging_freq=5)  # 0.892852
modellgb.fit(X_train, Y_train)
Y_pre_lgb = modellgb.predict(X_validation)
fpr, tpr, thresholds = roc_curve(Y_validation, Y_pre_lgb)
auc_lgb = roc_auc_score(Y_validation, Y_pre_lgb)
print('LightGBM模型的AUC值为：{:.6f}\n'.format(auc_lgb))
print('LightGBM耗时：{0}s\n'.format(round(time.time() - time_5, 2)))
print()
# 画ROC曲线图
plt.sca(ax1)
plt.xlim(-0.02, 1.02)
plt.ylim(0, 1.02)
plt.plot(fpr, tpr, 'b')
plt.title('$ROC curve-LGB$')
plt.plot([0, 1], [0, 1], 'r--')
plt.fill_between(fpr, tpr, color='lightgreen', alpha=0.6)
plt.text(0.1, 0.65, 'AUC = {:.4f}'.format(auc_lgb), fontsize=15)

# 搭建模型6(XGB) #0.003
time_6 = time.time()
modelxgb = xgboost.XGBClassifier(n_estimators=50, max_depth=5, colsample_bytree=0.6, colsample_bylevel=0.5,
                                 learning_rate=0.1)  # 889261
modelxgb.fit(X_train, Y_train)
Y_pre_xgb = modelxgb.predict_proba(X_validation)
Y_pre_xgb = np.array(Y_pre_xgb[:, 1])
fpr, tpr, thresholds = roc_curve(Y_validation, Y_pre_xgb)
auc_xgb = roc_auc_score(Y_validation, Y_pre_xgb)
print('XGB模型的AUC值为：{:.6f}\n'.format(auc_xgb))
print('XGB耗时：{0}s\n'.format(round(time.time() - time_6, 2)))
# 画ROC曲线图
plt.sca(ax2)
plt.xlim(-0.02, 1.02)
plt.ylim(0, 1.02)
plt.plot(fpr, tpr, 'b')
plt.title('$ROC curve-XGB$')
plt.plot([0, 1], [0, 1], 'r--')
plt.fill_between(fpr, tpr, color='lightgreen', alpha=0.6)
plt.text(0.1, 0.65, 'AUC = {:.4f}'.format(auc_xgb), fontsize=15)
plt.show()

# 八、模型预测及生成提交数据集(30s)
print('--------------------模型预测及生成提交数据集--------------------')
np.set_printoptions(suppress=True)  # 取消arr的科学计数法
time_5 = time.time()
# xgb_test = xgb.XGBClassifier()
xgb_test = xgboost.XGBClassifier(n_estimators=50, max_depth=5, colsample_bytree=0.6, colsample_bylevel=0.5,
                                 learning_rate=0.1)  # 889261
xgb_test.fit(final_train_dataset.loc[:, X_var].values, final_train_dataset['target'])
Y_pre_xgb_final = xgb_test.predict_proba(final_test_dataset.loc[:, X_var].values)
Y_pre_xgb_final = np.array(Y_pre_xgb_final[:, 1])
print('搭建模型耗时：{0}s\n'.format(round(time.time() - time_5, 2)))
# 生成提交数据集
submit_test_dataset['target'] = Y_pre_xgb_final
submit_test_dataset['Date_received'] = submit_test_dataset['Date_received'].apply(FT.to_int)
print('提交格式整理完毕\n')

time_6 = time.time()
submit_test_dataset.drop(['Merchant_id', 'Discount_rate', 'Distance'], axis=1, inplace=True)
submit_test_dataset.to_csv(r'E:\PycharmProjects\sklearn\Data\o2o_coupon\submission_dataset.csv', header=0, index=False)
print('生成提交数据集用时：{0}s\n'.format(round(time.time() - time_6, 2)))

print("DONE")
