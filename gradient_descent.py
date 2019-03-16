#! /Users/michael/anaconda3/bin/python
# @Date:   2019-03-16 10:41:49

"""
仿照 sklearn 的风格用自己的代码来实现梯度下降算法，从而创建线性回归模型
代码只是为了说明整个算法的思路，要确保代码保持尽可能简洁，因此没有过多处理异常
"""

import numpy as np 


class LinearRegression:

    def __init__(self, alpha=0.001, precision=5):
        self.alpha = alpha
        self.precision = precision

    def fit(self, X, y):
        X = np.c_[np.ones((X.shape[0],1)), X]  # 增加一维作为截距
        self.theta = np.zeros((X.shape[1], 1)) # 初始值为全零的参数向量
        count = 0 # 迭代次数计数器，是评估算法优劣的指标之一
        while True:
            count += 1   
            last_theta = np.copy(self.theta)
            error = np.dot(X, self.theta)-y
            gradient = np.dot(X.T, error)
            self.theta = self.theta - self.alpha * gradient
            if abs(self.theta - last_theta).mean() < 0.00000001:
                break
        print('迭代次数:', count)
        print('coef_ =', [round(i[0], self.precision) for i in self.theta[1:]])
        print('intercept_ =', round(self.theta[0][0], self.precision))

    def predict(self, x): # 传入变量向量，给出预测值
        x = np.array(x)
        # import ipdb; ipdb.set_trace()
        x = np.r_[[1], x] # 首位加一个1，用于和截距相乘
        return round(np.dot(x, self.theta)[0], self.precision)


class StandardScaler:

    def __init__(self):
        self.stds = []
        self.means = []

    def fit(self, X): # 计算每个特征的均值和标准差
        X = np.array(X) # 传入列表的话，自动转ndarray对象
        for col in X.T: # 转置后遍历列
            self.stds.append(col.std())
            self.means.append(col.mean())

    def transform(self, X): # 标准化每一列
        X = np.array(X)
        cols_trans = []
        for index, col in enumerate(X.T):
            cols_trans.append((col - self.means[index])/self.stds[index])
        return np.array(cols_trans).T

print('没有作标准化：')

X = np.array([[1, 3], [2, 4]])
y = np.array([[10], [15]])

lr1 = LinearRegression()
lr1.fit(X, y)
print(lr1.predict([1,3]))
print(lr1.predict([3,5]))

print('\n----华丽的分界线----\n')
print('经过标准化的训练集：')

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
lr2 = LinearRegression()
lr2.fit(X, y)
print(lr2.predict(scaler.transform([1, 3])))
print(lr2.predict(scaler.transform([3, 5])))

"""
运行结果如下：

没有作标准化：
迭代次数: 50947
coef_ = [2.49992, 2.50002]
intercept_ = 5e-05
10.00004
19.99991

----华丽的分界线----

经过标准化的训练集：
迭代次数: 6811
coef_ = [1.25, 1.25]
intercept_ = 12.49999
9.99999
19.99999

可以看到经过标准化之后的训练集能够更快收敛，提高训练效率。
"""