#!/bin/python
#-*-coding: utf-8 -*-

#lambda��һ�����ʽ

def f(x):
    return x ** 2
    
g = lambda x : x ** 2
   
if __name__ == '__main__':
    print f(4)
    print g(4)
    
'''
map����������ṩ�ĺ�����ָ��������ӳ�䡣map(function, sequence[, sequence, ...]) -> list
�ڲ������ڶ������ʱ����������ÿ����������ͬλ�õ�Ԫ������������function������
ע��function�����Ĳ���������Ҫ��map���ṩ�ļ���������ƥ�䡣
������ϳ��Ȳ���ȣ�������С���ȶ����м��Ͻ��н�ȡ��
������ΪNoneʱ��������zip����
'''
list = map(lambda x : x ** 2, [1,2,3,4,5])
print list

addlist = map(lambda x, y: x + y, [1,2,3], [4,5,6])
print addlist

nonelist = map(None, [1,2,3], [4,5,6])
print nonelist


'''
reduce function: reduce������Բ���������Ԫ�ؽ����ۻ�
reduce�����Ķ��壺reduce(function, sequence[, initial]) -> value
function������һ�������������ĺ�����reduce���δ�sequence��ȡһ��Ԫ�أ�����һ�ε���function�Ľ���������ٴε���function��
��һ�ε���functionʱ������ṩinitial����������sequence�еĵ�һ��Ԫ�غ�initial��Ϊ��������function�������������sequence�е�ǰ����Ԫ������������function��
'''
reducevalue1 = reduce(lambda x,y : x + y, [1,2,3,4,5], 1)
print reducevalue1

reducevalue2 = reduce(lambda x,y : x + y, [1,2,3,4,5])
print reducevalue2

'''
filter�������ָ������ִ�й��˲�����
filter�����Ķ��壺
filter(function or None, sequence) -> list, tuple, or string
function��һ��ν�ʺ���������һ�����������ز���ֵTrue��False��
filter����������в���sequence�е�ÿ��Ԫ�ص���function��������󷵻صĽ���������ý��ΪTrue��Ԫ�ء�
����ֵ�����ͺͲ���sequence��������ͬ
'''

filterlist = filter(lambda x : x &1 != 0, [1,2,3,4,5,6])
print filterlist
