
# res = issubclass(int, object)
# print (res)

# def foo():
# 	pass

# print (foo.__class__)

# print (issubclass(foo.__class__, object))

# def outer():
#      x = 1
#      def inner():
#          print x # 1
#      return inner
# foo = outer()
# foo()
# foo.func_closure # doctest: +ELLIPSIS

# def outer(x):
#      def inner():
#          print x # 1
#      return inner
# print1 = outer(1)
# print2 = outer(2)
# print1()
# print2()


# def outer(some_func):
#      def inner():
#          print "before some_func"
#          ret = some_func() # 1
#          return ret + 1
#      return inner
# def foo():
#      return 1
# decorated = outer(foo) # 2
# decorated()

# class allen(object):
# 	def __str__(self):
# 		return "allen __str__"
# 	def __repr__(self):
# 		return "allen__repr__"

# a = allen()

# print (a)       # allen__str__
# print (str(a))  # allen__str__
# print (repr(a)) # allen__repr__


# s1 = set([1,2,3,4,4])
# s2 = set([1,2,3,4,6,7,8,8])
# print (s1)
# print (s2)
# print (s1 & s2)
# print (s1 | s2)
# s1.add(10)
# print (s1)

# s1.remove(3)
# print (s1)

# r = range(100)
# print (r)

# from collections import Iterable

# print (isinstance("abc", Iterable))

# res = [x *x for x in range(10) if x % 2 == 0]
# print (res)

# class allen(object):
# 	__slots__ = ['num']
# 	def __init__(self):
# 		self.num = 0
#     def __call__(self, num = 1):
# 		self.num += num

# a = allen()
# print (a.num)
# a(2)
# print (a.num)
# a(3)
# print (a.num)
# a.count = 100
# print (a.count)


# class animal(object):
# 	def __init__(self, name):
# 		self.name = name
# 	def greet(self):
# 		print 'hello i am %s' % self.name

# class dog(animal):
# 	def greet(self):
# 		super(dog, self).greet()  #python 2
# 		#super.greet()            #python 3
# 		print ('wang wang...')

# ani = animal('trump')
# ani.greet()

# d = dog('obama')
# d.greet()

# print dog.mro()


# a = [1, 2, 3]
# b = a[:]
# print a == b and a is not b

# a = [1, 2, 3]
# b = map(lambda x: x**2, a)
# c = [x**2 for x in a]
# d = [x**2 for x in a if x % 2 == 0]
# #e = (x**2 for x in range(1, 10))
# print b
# print c
# print d
# #print e

# from random import randint

# def odd(x):
# 	return x % 2

# allnums = []
# for i in range(10):
# 	allnums.append(randint(1,100))
# print filter(lambda x: x % 2, allnums)
# print [x for x in allnums if x % 2]

# print map(lambda x,y: x+y, range(10), range(10))
# print zip(range(10), range(10))

#print reduce(lambda x,y: x+y, range(100))

# def foo():
# 	m = 3
# 	def bar():
# 		n = 4
# 		print m + n

# print m
# foo()

# def counter(x):
# 	count = [x]
# 	def incr():
# 		count[0] += 1
# 		return count[0]
# 	return incr

# count = counter(5)
# print count()
# print count()

# -*- coding: UTF-8 -*-

import traceback
from snack import *
 
screen = SnackScreen()
 
def window():
    btn1 = Button('button1')
    btn2 = Button('button2')
    g = Grid(2, 1)
    g.setField(btn1, 0, 0)
    g.setField(btn2, 1, 0)
    screen.gridWrappedWindow(g, "My GUI")
    f = Form()
    f.add(g)
    result = f.run()
    screen.popWindow()
 
 
def main():
    try:
        window()
    except:
        print traceback.format_exc()
    finally:
        screen.finish()
        return ''













