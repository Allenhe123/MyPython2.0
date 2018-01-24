
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

'''
# GaussianNB_高斯朴素贝叶斯, GaussianNB就是先验为高斯分布的朴素贝叶斯
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
clf = GaussianNB()
clf.fit(X, Y)
print (clf.predict([[-0.8, -1]]))

print(clf.predict_proba([[-0.8, -1]]))

clf_pf = GaussianNB()
clf_pf.partial_fit(X, Y, np.unique(Y))
print (clf_pf.predict([[-0.8, -1]]))
'''

'''
# MultinomialNB_多项朴素贝叶斯, MultinomialNB就是先验为多项式分布的朴素贝叶斯
X = np.random.randint(5, size=(6, 100)) # 每个int范围是[0-5)，size is shape: 6 * 10
y = np.array([1, 2, 3, 4, 5, 6])
clf = MultinomialNB()
clf.fit(X, y)
print (clf.predict(X[2:3]))
'''

# BernoulliNB_伯努利朴素贝叶斯  BernoulliNB就是先验为伯努利分布的朴素贝叶斯
X = np.random.randint(2, size=(6, 100))
Y = np.array([1, 2, 3, 4, 4, 5])
clf = BernoulliNB()
clf.fit(X, Y)
print(clf.predict(X[2:3]))



'''
这三个类适用的分类场景各不相同，一般来说，如果样本特征的分布大部分是连续值，使用GaussianNB会比较好。
如果如果样本特征的分大部分是多元离散值，使用MultinomialNB比较合适。
而如果样本特征是二元离散值或者很稀疏的多元离散值，应该使用BernoulliNB。

GaussianNB一个重要的功能是有 partial_fit方法，这个方法的一般用在如果训练集数据量非常大，一次不能全部载入内存的时候。
这时我们可以把训练集分成若干等分，重复调用partial_fit来一步步的学习训练集，非常方便

http://blog.csdn.net/abcd1f2/article/details/51249702

'''