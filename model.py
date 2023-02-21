# -*- coding: UTF-8 -*-
# 为方便测试，请统一使用 numpy、pandas、sklearn 三种包，如果实在有特殊需求，请单独跟助教沟通
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
import argparse
import math
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from sklearn import manifold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics as metrics
import shap

# 设定随机数种子，保证代码结果可复现
np.random.seed(1024)


class Model:
    """
    要求：
        1. 需要有__init__、train、predict三个方法，且方法的参数应与此样例相同
        2. 需要有self.X_train、self.y_train、self.X_test三个实例变量，请注意大小写
        3. 如果划分出验证集，请将实例变量命名为self.X_valid、self.y_valid
    """
    # 模型初始化，数据预处理，仅为示例
    def __init__(self, train_path, test_path):
        df_train = pd.read_csv(train_path, encoding='utf-8')
        df_test = pd.read_csv(test_path, encoding='utf-8')
        self.rfr = RandomForestRegressor(n_estimators=110, random_state=0)
        self.pca = PCA(n_components=10)
        self.sc = StandardScaler()
        self.df_train = df_train
        self.df_test = df_test
        
        self.train_data_raw = df_train

        # 初始化模型
        # self.regression_model = LinearRegression()
        self.regression_model = Ridge(alpha=0.5)
        self.df_predict = pd.DataFrame(index=df_test.index)

    # 模型训练，输出训练集平均绝对误差和均方误差
    def train(self):
        print(f'total train data:{self.X_train.shape[0]}')
        # self.rfr.fit(self.X_train, self.y_train)
        self.regression_model.fit(self.X_train, self.y_train)
        # y_train_pred = self.rfr.predict(self.X_train)
        y_train_pred = self.regression_model.predict(self.X_train)
        print(f'R^2:{r2_score(self.y_train,y_train_pred)}')
        return mean_absolute_error(self.y_train, y_train_pred), mean_squared_error(self.y_train, y_train_pred)

    # 模型测试，输出测试集预测结果
    def predict(self):
        # y_test_pred = self.rfr.predict(self.X_test)
        y_test_pred = self.regression_model.predict(self.X_test)
        self.df_predict['出售价格'] = y_test_pred
        return self.df_predict

    # 数据预处理
    def preprocess_train(self):
        train_df = self.train_data_raw.copy()
        # 缺失值处理
        train_df = train_df.replace([' -  ', '0', 0, np.nan], np.nan)

        train_df['总单元'] = train_df['总单元'].fillna(train_df['总单元'].mode()[0])
        train_df['居住单元'] = train_df['居住单元'].fillna(train_df['居住单元'].mode()[0])
        train_df['总平方英尺'] = train_df['总平方英尺'].astype('float64')
        # train_df['土地平方英尺'] = train_df['土地平方英尺'].astype('float64')
        train_df['总平方英尺'] = train_df['总平方英尺'].fillna(
            train_df.groupby('居住单元')['总平方英尺'].transform('mean'))
        # train_df['土地平方英尺'] = train_df['土地平方英尺'].fillna(train_df.groupby('居住单元')['土地平方英尺'].transform('mean'))
        train_df['总平方英尺'] = train_df['总平方英尺'].fillna(train_df['总平方英尺'].mode()[0])
        # train_df['土地平方英尺'] = train_df['土地平方英尺'].fillna(train_df['土地平方英尺'].mode()[0])
        train_df['邮编'] = train_df['邮编'].astype('float64')
        train_df['邮编'] = train_df['邮编'].fillna(train_df['邮编'].mode()[0])
        train_df['修建年份'] = train_df['修建年份'].astype('float64')
        train_df['修建年份'] = train_df['修建年份'].fillna(train_df['修建年份'].mode()[0])
        train_df['出售价格'] = train_df['出售价格'].astype('float64')
        train_df['出售价格'] = train_df['出售价格'].fillna(0)

        # 效果不好，不应该删除出售价格较大的数据，
        # train_df = train_df[train_df['出售价格'] >= 100]
        # train_df = train_df[train_df['出售价格'] <= 5000000000]
        # train_df['出售价格'] = train_df['出售价格'].apply(lambda x: 100 if x < 100 else x)
        # train_df['出售价格'] = train_df['出售价格'].apply(lambda x: 5000000000 if x > 5000000000 else x)

        train_df['公寓号'] = train_df['公寓号'].fillna(train_df['公寓号'].mode()[0])
        train_df = train_df.drop(train_df[train_df['出售价格'] == 0].index)
        train_df = train_df.reset_index()

        self.y_train = train_df['出售价格']
        # 关于为什么删掉出售日期的解释:解释性很差
        train_df = train_df.drop(columns=['商业单元', '出售价格', '土地平方英尺', 'index', '地役权', '出售日期'])

        num_cols = [i for i in train_df.columns if train_df[i].dtype in ['int64', 'float32', 'float64']]
        cat_cols = [i for i in train_df.columns if train_df[i].dtype == 'object']
        self.num_cols = num_cols

        te_cols = []
        for col in num_cols:
            self.target_encoding(col, train_df, cat_cols, te_cols)
        # one-hot encodings(not used)
        for col in cat_cols:
            dummies = pd.get_dummies(train_df[col], prefix=col).astype('int32')
            train_df = pd.concat([train_df, dummies], axis=1)
        train_df = train_df.drop(columns=cat_cols)
        self.X_train = train_df

        # 可视化预处理后的数据
        X_tsne = manifold.TSNE(n_components=2, random_state=42, verbose=2, init='pca').fit_transform(train_df)
        plt.figure(figsize=(13, 10))
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="jet")
        plt.axis('off')
        plt.colorbar()
        plt.show()

    def target_encoding(self,target_feature, df, cat_cols, te_cols):
        if len(cat_cols) == 0:
            return 'All features are encoded'
        target_std = df[target_feature].std()
        for col in cat_cols:
            encoded_std = df.groupby(col)[target_feature].mean().std()
            if target_std < encoded_std:
                df[col] = df.groupby(col)[target_feature].transform('mean')
                cat_cols.remove(col)
                te_cols.append(col)

    def preprocess_test(self):
        test_df = self.df_test.copy()
        test_df = test_df.replace([' -  ', ' -  ', '0', 0, np.nan], np.nan)
        missing = self.df_train.isnull().sum()[self.df_train.isnull().sum() > 0]
        missing_df = pd.DataFrame({'NaN_count': missing, 'NaN_percentage': missing / len(self.df_train)}).sort_values(
            by='NaN_percentage', ascending=False)

        # %%
        test_df['总单元'] = test_df['总单元'].fillna(test_df['总单元'].mode()[0])
        test_df['居住单元'] = test_df['居住单元'].fillna(test_df['居住单元'].mode()[0])
        test_df['总平方英尺'] = test_df['总平方英尺'].astype('float64')
        test_df['总平方英尺'] = test_df['总平方英尺'].fillna(
            test_df.groupby('居住单元')['总平方英尺'].transform('mean'))
        test_df['总平方英尺'] = test_df['总平方英尺'].fillna(test_df['总平方英尺'].mode()[0])

        test_df['邮编'] = test_df['邮编'].astype('float64')
        test_df['邮编'] = test_df['邮编'].fillna(test_df['邮编'].mode()[0])
        test_df['修建年份'] = test_df['修建年份'].astype('float64')
        test_df['修建年份'] = test_df['修建年份'].fillna(test_df['修建年份'].mode()[0])
        
        test_df['公寓号'] = test_df['公寓号'].fillna(test_df['公寓号'].mode()[0])
        test_df = test_df.reset_index()

        # 删去解释性较差的特征
        test_df = test_df.drop(columns=['商业单元', '土地平方英尺', 'index', '地役权'])

        test_df.drop(columns=['出售日期'],inplace=True)
        num_cols = [i for i in test_df.columns if test_df[i].dtype in ['int64', 'float32', 'float64']]
        cat_cols = [i for i in test_df.columns if test_df[i].dtype == 'object']

        te_cols = []
        for col in num_cols:
            self.target_encoding(col, test_df, cat_cols, te_cols)

        for col in cat_cols:
            dummies = pd.get_dummies(test_df[col], prefix=col).astype('int32')
            test_df = pd.concat([test_df, dummies], axis=1)
        test_df = test_df.drop(columns=cat_cols)
        
        X_pre = test_df.copy()
        # X_pre = self.sc.fit_transform(X_pre)
        # X_pre = self.pca.fit_transform(X_pre)
        self.X_test = X_pre
        

    def self_eval(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state= 42 )
        # 标准化
        X_train = self.sc.fit_transform(X_train)
        X_test = self.sc.transform(X_test)

        #提取主成分

        X_train = self.pca.fit_transform(X_train)
        X_test = self.pca.transform(X_test)
        self.plot_variance()
        plt.show()

        self.rfr.fit(X_train, y_train)
        y_pred = self.rfr.predict(X_test)
        self.regression_model.fit(X_train, y_train)
        y_pred = self.regression_model.predict(X_test)
        print("RMSE:", metrics.mean_squared_error(y_test, y_pred) ** 0.5)
        
        # 使用shap库可视化
        X = self.X_train[self.num_cols].copy()
        shap_df = pd.concat([self.X_train, X], axis=1)
        X_shap, X_val, y_shap, y_val = train_test_split(shap_df, self.y_train, train_size=0.8, random_state=0)
        shap_model = RandomForestRegressor(random_state=0).fit(X_shap, y_shap)
        explainer = shap.TreeExplainer(shap_model)
        shap_values = explainer.shap_values(X_val)
        # %%
        shap.summary_plot(shap_values, X_val, plot_size=(16, 8))

    # 可视化特征之间的相关性
    def plot_variance(self,width=8, dpi=100):
        # create figure
        fig, axs = plt.subplots(1, 2)
        n = self.pca.n_components_
        grid = np.arange(1, n + 1)
        # Explained variance
        evr = self.pca.explained_variance_ratio_
        axs[0].bar(grid, evr)
        axs[0].set(xlabel='Component', title='% Explained Variance', ylim=(0.0, 1.0))
        # Cumulative Variance
        cv = np.cumsum(evr)
        axs[1].plot(np.r_[0, grid], np.r_[0, cv], 'o-')
        axs[1].set(xlabel='Component', title='% Cumulative Variance', ylim=(0.0, 1.0))
        # Set Up Figure
        fig.set(figwidth=12, dpi=100)
        return axs


# 以下部分请勿改动！
if __name__ == '__main__':
    # 解析输入参数。在终端执行以下语句即可运行此代码： python model.py --train_path "./data/train.csv" --test_path "./data/test.csv"
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="train.csv", help="path to train dataset")
    parser.add_argument("--test_path", type=str, default="test.csv", help="path to test dataset")
    opt = parser.parse_args()

    model = Model(opt.train_path, opt.test_path)
    model.preprocess_train()
    model.preprocess_test()
  
    print(f'训练集维度:{model.X_train.shape}\n测试集维度:{model.X_test.shape}')
    mae_score, mse_score = model.train()
    print(f'mae_score={mae_score:.6f}, mse_score={mse_score:.6f}, mrse_score={math.sqrt(mse_score):.6f}')
    predict = model.predict()
    predict.to_csv('200110526_李萌_submit_33.csv', index=False, encoding='utf-8-sig')
