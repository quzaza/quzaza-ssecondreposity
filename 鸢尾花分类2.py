import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn import svm



def iris_type(s):
    
    it = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2} 
    b = s.decode("gbk")
    return it[b]



path = 'iris.data'  # 数据文件路径

data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})



x, y = np.split(data, (4,), axis=1)

x = x[:, :2]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.6)



# clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')

clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')

clf.fit(x_train, y_train.ravel())



print(clf.score(x_train, y_train))  # 精度

y_hat = clf.predict(x_train)

accuracy_score(y_hat, y_train, '训练集')

print(clf.score(x_test, y_test))

y_hat = clf.predict(x_test)

accuracy_score(y_hat, y_test, '测试集')



print('decision_function:\n', clf.decision_function(x_train))

print('\npredict:\n', clf.predict(x_train))



x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0列的范围

x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1列的范围

x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  # 生成网格采样点

grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点

print('grid_test = \n', grid_test)

grid_hat = clf.predict(grid_test)       # 预测分类值

grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同



mpl.rcParams['font.sans-serif'] = ['SimHei']

mpl.rcParams['axes.unicode_minus'] = False

cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])

cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])

plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)

plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)

plt.scatter(x[:, 0], x[:, 1], c='k', edgecolors='k', s=50, cmap=cm_dark)  # 样本

plt.scatter(x_test[:, 0], x_test[:, 1], s=120, facecolors='none', zorder=10)  # 圈中测试集样本

plt.xlabel('花萼长度', fontsize=13)

plt.ylabel('花萼宽度', fontsize=13)

plt.xlim(x1_min, x1_max)

plt.ylim(x2_min, x2_max)

plt.title('鸢尾花SVM二特征分类', fontsize=15)

# plt.grid()

plt.show()