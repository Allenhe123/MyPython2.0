
# import os

# print "hello world"

# x = 10
# def foo():
#     y = 5
#     bar = lambda : x + y
#     print (bar())

# foo()

# j, k = 1, 2

# def proc1():
#     j, k = 3, 4
#     print ('j=%d, k=%d', (j, k))
#     k = 5

# def proc2():
#     j = 6
#     proc1()
#     print "j = %d, k = %d", (j, k)

# k = 7
# proc1() #3,4
# print "j = %d, k = %d", (j, k) #3,7

# j = 8
# proc2()   #3,4  6,7
# print "j = %d, k = %d", (j, k)  #8,7

# def foo(n):
#     if n ==0 or n == 1 : 
#         return 1
#     else :
#         return n * foo(n - 1)
# print foo(10)

# from random import randint
# def randGen(alist):
#     while (len(alist) > 0):
#         print 'alist size = ', len(alist)
#         yield alist.pop(randint(0, len(alist) - 1))

# blist = [1,2,3,4,5]
# gen = randGen(blist)
# for i in range(len(blist)):
#     print gen.next()


# class C(object):
#     def __init__(self):
#         print ('create an instance of ', self.__class__.__name__)

# class P(C):
#     pass

# c = C()
# p = P()
# print(c.__class__.__name__)
# print(C.__bases__)
# print(p.__class__.__name__)
# print(P.__bases__)
        

# class P(object):
#     def foo(self):
#         print('P - foo')

# class C(P):
#     pass


# p = P()
# c = C()
# p.foo()
# c.foo()

# b = issubclass(C, P)
# print(b)


# class RoundFloat(object):
#     def __init__(self, val):
#         assert(isinstance(val, float))
#         self.value = round(val, 2)

#     def __str__(self):
#         # return str(self.value)
#         return '%.2f' % self.value
#     __repr__ = __str__



# rf = RoundFloat(4.5964)
# print rf



def allen():
    '''
    for i in range(100000):
        if (i % 9 == 0) and (i % 8 == 1) and (i % 7 == 0) and (i % 6 == 3) and (i % 5 == 4) and (i % 4 == 1) and (i % 3 == 0) and (i % 2 == 1) :
            print i

    import numpy

    a = numpy.tile([0, 0], 5)
    print(a)

    b = numpy.tile([0, 0], (1,1))
    print(b)

    c = numpy.tile([0, 0], (2,1))
    print(c)
    '''

    import numpy as np
    y = np.array([3, 0, 2, 1, 4, 5])
    indexarray = y.argsort()
    print(indexarray)

if __name__ == '__main__':
    allen()







