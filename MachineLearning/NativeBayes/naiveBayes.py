import numpy as np
import pandas as pd
from functools import reduce

def get_word_set(postingList):
    word_set = set([])
    for i in range(len(postingList)):
        s = set(postingList[i])
        # 集合update方法：是把要传入的元素拆分，做为个体传入到集合中. 主要用来消除重复元素.
        word_set.update(s)
    return list(word_set)

def get_word_matrix(word_set, postingList):
    data_word = np.zeros((len(postingList), len(word_set))) # 6 x 32
    for i in range(len(postingList)):
        count_word = [postingList[i].count(s) for s in word_set]   # 列表生成式来产生列表
        data_word[i] = count_word
    data_word = pd.DataFrame(data_word, columns = word_set)
    return data_word

def get_p(data_word, classVec):
    class_set = list(set(classVec))  # [0, 1]
    len_class = len(class_set)       # 2
    p_matrix = np.zeros((len_class, data_word.shape[1]))  # 2 x 32
    for i in range(len_class):
        p_matrix[i] =  np.sum(data_word[np.array(classVec) == class_set[i] ], axis =0) / np.sum(data_word, axis=0)
    return pd.DataFrame(p_matrix,index = class_set,columns =data_word.columns)


def get_class(testEntry, p_matrix, classVec):
    class_set = list(set(classVec))
    p = -1
    for i in range(len(class_set)):
        p_i = float(classVec.count(class_set[i])) / len(classVec)
        p_word = [p_matrix.loc[class_set[i], word] for word in testEntry if word in word_set]
        p_i_word = p_i * reduce(lambda x, y : x * y, p_word)
        print('class: %s p_i_word %s' % (class_set[i], p_i_word))
        if p_i_word > p :
           p = p_i_word
           class_predict = class_set[i]
    return p, class_predict

if __name__ == '__main__':
   postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

   classVec = [0, 1, 0, 1, 0, 1]
   word_set = get_word_set(postingList)  # 消除重复后的 32 words list

   data_word = get_word_matrix(word_set, postingList)

   p_matrix = get_p(data_word, classVec)

   testEntry = ['love', 'my', 'dalmation']

   p, class_predict = get_class(testEntry, p_matrix, classVec)

   print ('p:',p)
   print ('class_predict:',class_predict)





   '''
   range([start, stop, step])，根据start与stop指定的范围以及step设定的步长，生成一个序列。
   xrange 用法与 range 完全相同，所不同的是生成的不是一个list对象，而是一个生成器
   所以xrange做循环的性能比range好，尤其是返回很大的时候。尽量用xrange吧，除非你是要返回一个列表。
   
   set是一个无序且不重复的元素集合。
   s = set()
   s = {11,22,33,44}  #注意在创建空集合的时候只能使用s=set()，因为s={}创建的是空字典
   
   在Python 3中，range()的实现方式与xrange()函数相同，所以就不存在专用的xrange()（在Python 3中使用xrange()会触发NameError）。
   
   '''