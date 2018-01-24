from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import numpy as np
import operator

def loadFile(filename):
    file = open(filename, "r")
    lines = file.readlines()
    sample = []
    lable = []

    for line in lines :
        input = str.split(line, ",")
        lst = []
        index = 0
        for s in input:
            index += 1
            if (index) > 16:
                num = int(s)
                lable.append(num)
                break
            else:
                #lst.append(float(s) / 100.0)
                lst.append(int(s))
        sample.append(lst)
    file.close()
    return sample, lable

# KNN Classifier
def knn_classifier(train_x, train_y):
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model

def predict_result(result_label, test_label):
    # first method
    count = 0
    print('Total count: ', len(result_label))
    zip_result = zip(result_label, test_label)
    for v1, v2 in zip_result:
        if v1 == v2:
            count += 1
    print('Match count: ', count)
    print('Rate: ', float(count) / float(len(result_label)))

    # second method
    accuracy = metrics.accuracy_score(test_label, result_label)
    print('accuracy: %.2f%%' % (100 * accuracy))

def predict(trainfile, testfile):
    sample, label = loadFile(trainfile)

    model = knn_classifier(sample, label)

    testsample, testlabel = loadFile(testfile)

    result_label = model.predict(testsample)

    predict_result(result_label, testlabel)


def predict_2(trainfile, testfile, K):
    sample, label = loadFile(trainfile)
    testsample, testlabel = loadFile(testfile)

    samplearray = np.array(sample)
    print(samplearray.shape[0])
    print(samplearray.shape[1])

    result_label = []
    for lst1 in testsample:
        # 计算距离
        arraytest = np.tile(lst1, (samplearray.shape[0], 1))
        diffmat = arraytest - samplearray
        sqdiffmat = np.square(diffmat)
        distanceArray = sqdiffmat.sum(axis=1)
        distance = np.sqrt(distanceArray)
        sortedDistIndicies = distance.argsort()

        # 选择距离最小的K个点
        classCount = {}
        for i in range(K):
            # 找到该样本的类型
            voteIlabel = label[sortedDistIndicies[i]]
            # 在字典中将该类型加1
            classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

        # 排序并返回出现最多的那个类型
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

        result_label.append(sortedClassCount[0][0])

    print(result_label)

    predict_result(result_label, testlabel)

## 增加数据的归一化

def autoNorm(dataSet):
    """
    归一化特征值，消除属性之间量级不同导致的影响
    :param dataSet: 数据集
    :return: 归一化后的数据集normDataSet,ranges和minVals即最小值与范围，并没有用到
    归一化公式：
        Y = (X-Xmin)/(Xmax-Xmin)
        其中的 min 和 max 分别是数据集中的最小特征值和最大特征值。该函数可以自动将数字特征值转化为0到1的区间。

    import numpy as np
    a = np.array([[1,5,3],[4,2,6]])
    print(a.min()) #无参，所有中的最小值
    print(a.min(0)) # axis=0; 每列的最小值
    print(a.min(1)) # axis=1；每行的最小值
    结果：
    1
    [1 2 3]
    [1 2]
    """
    # 计算每种属性的最大值、最小值、范围
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    # 极差
    ranges = maxVals - minVals
    # -------第一种实现方式---start-------------------------
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    # 生成与最小值之差组成的矩阵
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    # 将最小值之差除以范围组成矩阵
    normDataSet = normDataSet / np.tile(ranges, (m, 1))  # element wise divide
    return normDataSet, ranges, minVals

def predict_3(trainfile, testfile, K):
    sample, label = loadFile(trainfile)
    testsample, testlabel = loadFile(testfile)

    # 转成nump的array
    samplearray = np.array(sample)
    testsample = np.array(testsample)

    # 归一化
    samplearray, rangesample, minval = autoNorm(samplearray)
    testsample, rangetest, minvaltest = autoNorm(testsample)
    print(samplearray.shape[0])
    print(samplearray.shape[1])
    print(testsample.shape[0])
    print(testsample.shape[1])

    result_label = []
    for lst1 in testsample:
        # 计算距离
        arraytest = np.tile(lst1, (samplearray.shape[0], 1))
        diffmat = arraytest - samplearray
        sqdiffmat = np.square(diffmat)
        distanceArray = sqdiffmat.sum(axis=1)
        distance = np.sqrt(distanceArray)
        sortedDistIndicies = distance.argsort()

        # 选择距离最小的K个点
        classCount = {}
        for i in range(K):
            # 找到该样本的类型
            voteIlabel = label[sortedDistIndicies[i]]
            # 在字典中将该类型加1
            classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

        # 排序并返回出现最多的那个类型
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        result_label.append(sortedClassCount[0][0])

    print(result_label)
    predict_result(result_label, testlabel)

if __name__ ==  '__main__':
    # predict("Train.dat", "Test.dat")       # 97.77%
    # predict_2("Train.dat", "Test.dat", 5)  # 97.80%
     predict_3("Train.dat", "Test.dat", 5)   # 97.80%