import math
import random
import time
import numpy as np
import matplotlib.pyplot as plt

#random.seed(0)

######################################
TEST_MODE = False
PLOT = False
TRAIN_FILE =  "Train.dat"
TEST_FILE =   "sample.log"
RESULT_FILE = "result.dat"
HIDDEN_LAYER_NUM = 15
OUTPUT_LAYER_NUM = 10
######################################

def rand(a, b):  # 随机函数
    return (b - a) * random.random() + a


def make_matrix(m, n, fill=0.0):  # 创建一个指定大小的矩阵
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return mat


# 定义sigmoid函数和它的导数
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivate(x):
    return x * (1 - x)  # sigmoid函数的导数

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

def LoadFile(filename):
    file = open(filename, "r")
    lines = file.readlines()
    sample = []
    lable = [];

    for line in lines :
        input = str.split(line, ",")
        lst = []
        single_label = []
        num = 0
        index = 0
        for s in input:
            index += 1
            if (index) > 16:
                num = int(s)
                #single_label.append(float(num) / 100.0)
                single_label.append(int(num))
                break
            else:
                lst.append(float(s) / 100.0)
        sample.append(lst)

        temp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] #13
        temp[single_label[0]] = 1.0
        lable.append(temp)
    file.close()
    return sample, lable

def LoadTestFile(filename):
    file = open(filename, "r")
    lines = file.readlines()
    sample = []

    for line in lines :
        input = str.split(line, ",")
        lst = []
        for s in input:
            lst.append(float(s) / 100.0)
        sample.append(lst)
    return sample

def write_result(inputname, outputname, result):
    filein = open(inputname, 'r')
    lines = filein.readlines()

    fileout = open(outputname, 'w')

    newresult = []
    length = len(result)
    index = 0
    for line in lines:
        ret = result[index]
        line = line.strip('\n')
        newline = line + ", " + str(ret) + "\n"
        print(newline)
        index +=1
        newresult.append(newline)
    for strs in newresult:
        fileout.write(strs)

    filein.close()
    fileout.close()

def find_lable(inputs):
    labels = []
    for arr in inputs:
        index = 0
        maxval = 0.0
        maxindex = 0
        for val in arr:
            if (val > maxval):
                maxval = val
                maxindex = index
            index += 1
        labels.append(maxindex)
    return  labels

def setupFigure(maxnum):
    plt.figure(figsize=(8, 5))
    plt.grid(True)  # 添加网格
    plt.ion()  # interactive mode on
    plt.xlabel("times")  # X轴标签
    plt.ylabel("error")  # Y轴标签
    plt.title("train error evolution")  # 标题
    plt.ylim(0, 2.5)  # Y轴的范围
    plt.xlim(0, maxnum)
    return plt


def figure(plt, index, error):
    x = index
    y = error
    plt.scatter(x, y, label="$error$", color="red")
    # plt.plot(x,y, "b--", label="$error$", color="red", linewidth=0.1)
    plt.pause(0.001)
    #plt.legend()
    #plt.show()
    #plt.savefig("easyplot.png")

class BPNeuralNetwork:
    def __init__(self):  # 初始化变量
        self.input_n = 0
        self.hidden_n = 0
        self.output_n = 0
        self.input_cells = []
        self.hidden_cells = []
        self.output_cells = []
        self.input_weights = []
        self.output_weights = []
        self.input_correction = []
        self.output_correction = []
        # 三个列表维护：输入层，隐含层，输出层神经元

    def setup(self, ni, nh, no):
        self.input_n = ni + 1  # 输入层+偏置项
        self.hidden_n = nh  # 隐含层
        self.output_n = no  # 输出层

        # 初始化神经元
        self.input_cells = [1.0] * self.input_n
        self.hidden_cells = [1.0] * self.hidden_n
        self.output_cells = [1.0] * self.output_n

        # 初始化连接边的边权
        self.input_weights = make_matrix(self.input_n, self.hidden_n)  # 邻接矩阵存边权：输入层->隐藏层
        self.output_weights = make_matrix(self.hidden_n, self.output_n)  # 邻接矩阵存边权：隐藏层->输出层

        # 随机初始化边权：为了反向传导做准备--->随机初始化的目的是使对称失效
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                self.input_weights[i][h] = rand(-0.2, 0.2)  # 由输入层第i个元素到隐藏层第j个元素的边权为随机值

        for h in range(self.hidden_n):
            for o in range(self.output_n):
                self.output_weights[h][o] = rand(-2.0, 2.0)  # 由隐藏层第i个元素到输出层第j个元素的边权为随机值
        # 保存校正矩阵，为了以后误差做调整
        self.input_correction = make_matrix(self.input_n, self.hidden_n)
        self.output_correction = make_matrix(self.hidden_n, self.output_n)

        # 输出预测值

    def predict(self, inputs):
        # 对输入层进行操作转化样本
        for i in range(self.input_n - 1):
            self.input_cells[i] = inputs[i]  # n个样本从0~n-1
        # 计算隐藏层的输出，每个节点最终的输出值就是权值*节点值的加权和
        for j in range(self.hidden_n):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_cells[i] * self.input_weights[i][j]
                # 此处为何是先i再j，以隐含层节点做大循环，输入样本为小循环，是为了每一个隐藏节点计算一个输出值，传输到下一层
            self.hidden_cells[j] = sigmoid(total)  # 此节点的输出是前一层所有输入点和到该点之间的权值加权和

        for k in range(self.output_n):
            total = 0.0
            for j in range(self.hidden_n):
                total += self.hidden_cells[j] * self.output_weights[j][k]
            self.output_cells[k] = sigmoid(total)  # 获取输出层每个元素的值
        return self.output_cells[:]  # 最后输出层的结果返回

    # 反向传播算法：调用预测函数，根据反向传播获取权重后前向预测，将结果与实际结果返回比较误差
    def back_propagate(self, case, label, learn, correct):
        # 对输入样本做预测
        self.predict(case)  # 对实例进行预测
        output_deltas = [0.0] * self.output_n  # 初始化矩阵
        for o in range(self.output_n):
            error = label[o] - self.output_cells[o]  # 正确结果和预测结果的误差：0,1，-1
            output_deltas[o] = sigmoid_derivate(self.output_cells[o]) * error  # 误差稳定在0~1内

        # 隐含层误差
        hidden_deltas = [0.0] * self.hidden_n
        for h in range(self.hidden_n):
            error = 0.0
            for o in range(self.output_n):
                error += output_deltas[o] * self.output_weights[h][o]
            hidden_deltas[h] = sigmoid_derivate(self.hidden_cells[h]) * error
            # 反向传播算法求W
        # 更新隐藏层->输出权重
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                change = output_deltas[o] * self.hidden_cells[h]
                # 调整权重：上一层每个节点的权重学习*变化+矫正率
                self.output_weights[h][o] += learn * change + correct * self.output_correction[h][o]
                # 更新输入->隐藏层的权重
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                change = hidden_deltas[h] * self.input_cells[i]
                self.input_weights[i][h] += learn * change + correct * self.input_correction[i][h]
                self.input_correction[i][h] = change
                # 获取全局误差
        error = 0.0
        for o in range(len(label)):
            error = 0.5 * (label[o] - self.output_cells[o]) ** 2  # 平方误差函数
        return error


    def train(self, cases, labels, limit=1000, learn=0.05, correct=0.1):
        print("training iteration times: ", limit)
        iter_index = 1
        cost_all_time =0

        if PLOT == True:
            plt = setupFigure(limit)

        for i in range(limit):  # 设置迭代次数
            print("training iteration index: ", iter_index)
            start_time = time.time()
            error = 0.0
            for j in range(len(cases)):  # 对输入层进行访问
                label = labels[j]
                case = cases[j]
                error += self.back_propagate(case, label, learn, correct)  # 样例，标签，学习率，正确阈值
            iter_index += 1
            end_time = time.time()
            delta_time = end_time - start_time
            cost_all_time += delta_time

            if PLOT == True:
                figure(plt, i, error)

            print("training cost time: ", cost_all_time)
            print("training error: ", error)

    def saveWeights(self, inputweights, hiddenweights):
        np.savetxt(inputweights, self.input_weights)
        np.savetxt(hiddenweights, self.output_weights)

    def loadWeights(self, inputweights, hiddenweights):
        a = np.loadtxt(inputweights, dtype=np.float)
        b = np.loadtxt(hiddenweights, dtype=np.float)
        return a, b

    def initWeights(self, inputwt, outputwt):
        self.input_weights = inputwt
        self.output_weights = outputwt

    def test(self):
        cases, labels = LoadFile(TRAIN_FILE)
        print("train cases: ", cases)
        print("train labels: ", labels)

        sample_num = len(cases)
        input_cell_num = len(cases[0])
        print("input sample num: ", sample_num)
        print("input cell num: ", input_cell_num)

        self.setup(input_cell_num, HIDDEN_LAYER_NUM, len(labels[0]))             # 初始化神经网络：输入层，隐藏层，输出层元素个数
        print("Neural Network training....")
        starttime = time.time()
        self.train(cases, labels, 100, 0.05, 0.1)  ##################### 可以更改
        endtime = time.time()
        print("training cost all time: ", endtime - starttime)
        print("Neural Network train end")

        self.saveWeights("inputweights.dat", "outputweights.dat")

        test_cases, test_labels = LoadFile("test.dat")
        print("test sample: ", test_cases)
        print("test_label: ", test_labels)

        print("begin to predict ...")
        result = []
        for case in test_cases:
            predict_result = self.predict(case)
            softmax_result = softmax(predict_result)
            result.append(softmax_result)
            #result.append(predict_result)

        #print("predict output: ", result)

        real_labels = find_lable(test_labels)
        predict_labels = find_lable(result)
        print("reallll labels: ", real_labels)
        print("predict labels: ", predict_labels)

        length = len(real_labels)
        index = 0
        num = 0
        while index < length:
            if real_labels[index] == predict_labels[index]:
                num += 1
            index += 1

        print("right_num: ", num)
        print("all_num: ", length)
        print("predict rate: ", float(num) / float(length))

if __name__ == '__main__':
    nn = BPNeuralNetwork()
    if TEST_MODE == True:
        nn.test()
    else:
        test_cases = LoadTestFile(TEST_FILE)
        input_cell_num = len(test_cases[0])
        #print(input_cell_num)
        nn.setup(input_cell_num, HIDDEN_LAYER_NUM, OUTPUT_LAYER_NUM)

        a, b = nn.loadWeights("inputweights.dat", "outputweights.dat")
        nn.initWeights(a, b)

        print("begin to predict ...")
        result = []
        for case in test_cases:
            predict_result = nn.predict(case)
            #print(predict_result)
            softmax_result = softmax(predict_result)
            result.append(softmax_result)
        predict_labels = find_lable(result)
        print("predict labels: ", predict_labels)

        write_result(TEST_FILE, RESULT_FILE, predict_labels)

        print("predict done, the result file is: ", RESULT_FILE)
        print("HOLY SHIT!!!")

'''
http://blog.csdn.net/Augster/article/details/52837474?locationNum=2&fps=1
'''
