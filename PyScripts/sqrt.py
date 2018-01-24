# -*- coding: UTF-8 -*-
__author__ = '931317255@qq.com'

def sqrt(v, t):
    u = v
    r = v
    while 1:
        m = u / 2
        if m ** 2 > v:
            u /= 2
        elif m * m == v:
            return m
        else:
            r = m
            break
    while 1:
        if abs(r ** 2 - v) < t:
            return r
        else:
            r += 0.01

print (sqrt(225, 0.1))

# 二分查找法
# 牛顿迭代法
# 梯度下降法(最速下降法)
# 泰勒公式展开等等
# 辗转相除法
# 卡马克算法
# http://blog.jobbole.com/109249/