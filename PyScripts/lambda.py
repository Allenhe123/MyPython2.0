#!/bin/python
#-*-coding: utf-8 -*-

#lambda是一个表达式

def f(x):
    return x ** 2
    
g = lambda x : x ** 2
   
if __name__ == '__main__':
    print f(4)
    print g(4)
    
'''
map函数会根据提供的函数对指定序列做映射。map(function, sequence[, sequence, ...]) -> list
在参数存在多个序列时，会依次以每个序列中相同位置的元素做参数调用function函数。
注意function函数的参数数量，要和map中提供的集合数量相匹配。
如果集合长度不相等，会以最小长度对所有集合进行截取。
当函数为None时，操作和zip相似
'''
list = map(lambda x : x ** 2, [1,2,3,4,5])
print list

addlist = map(lambda x, y: x + y, [1,2,3], [4,5,6])
print addlist

nonelist = map(None, [1,2,3], [4,5,6])
print nonelist


'''
reduce function: reduce函数会对参数序列中元素进行累积
reduce函数的定义：reduce(function, sequence[, initial]) -> value
function参数是一个有两个参数的函数，reduce依次从sequence中取一个元素，和上一次调用function的结果做参数再次调用function。
第一次调用function时，如果提供initial参数，会以sequence中的第一个元素和initial作为参数调用function，否则会以序列sequence中的前两个元素做参数调用function。
'''
reducevalue1 = reduce(lambda x,y : x + y, [1,2,3,4,5], 1)
print reducevalue1

reducevalue2 = reduce(lambda x,y : x + y, [1,2,3,4,5])
print reducevalue2

'''
filter函数会对指定序列执行过滤操作。
filter函数的定义：
filter(function or None, sequence) -> list, tuple, or string
function是一个谓词函数，接受一个参数，返回布尔值True或False。
filter函数会对序列参数sequence中的每个元素调用function函数，最后返回的结果包含调用结果为True的元素。
返回值的类型和参数sequence的类型相同
'''

filterlist = filter(lambda x : x &1 != 0, [1,2,3,4,5,6])
print filterlist
