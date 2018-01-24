# !/usr/bin/python

import numpy as np

a = np.array([[1,2,3,4], [5,6,7,8]])
print a
print a.ndim
print a.shape
print a.dtype

b = np.zeros((2,3))
print b

c = np.ones((2,3))
print c

d = np.empty((2,3))
print d 

e = np.arange(1,20,5)
print e 

f = np.linspace(0,2,10)
print f 

print a[0]
print a[1]

print a[:, 0]
print a[:, 1]

print a[1, 2]

###################

m = np.array([1,2,3,4])
n = np.array([5,6,7,8])
print m + n

print m - n

print m * n

print m ** n 

print m / n 

print np.dot(m, n)

o = m > 2
print o 

print m.max()
print m.min()
print m.sum()

p = a.copy()
print p 

q = np.matrix([[1,2], [3,4]])
print q 

print type(q)
print q.T 

r = np.matrix([[4,5], [6,7]])
print r 

print q * r

print q.I 

#s = np.solve(r, q)


x = np.float32(np.random.rand(2, 10))
print 'x: '
print x

print 'y: '
y = np.dot([0.100, 0.200], x)
print y 

############################################

mymat = np.mat(a)
print type(mymat)
print mymat

##
eyemat = np.eye(4)
print eyemat

#
z = a.tolist()
print type(z)
print z

##############################################################

data = np.arange(15).reshape(3,5)
print data
print data.ndim
print data.shape
print data.size
print data.dtype
print data.itemsize

print data + 2
print data * 2
print data / 2
print data - 2

int16_array = np.array([1,2,3,4], dtype=np.int16)
print int16_array

float64_array = np.array([1,2,3,4], dtype=np.float64)
print float64_array

#default  is float64
random_array = np.empty((3,4), dtype=np.float64)
print random_array

arr = np.arange(-3, 3, 1)
print arr

print np.abs(arr)
print np.sqrt(np.abs(arr))
print np.square(arr)
print np.exp(arr)
print np.log(np.abs(arr))
print np.sign(arr)
print np.cos(arr)
print np.sin(arr)

arr = np.arange(6).reshape(2,3)
print arr
print np.sum(arr)
print np.sum(arr, axis = 1)
print np.sum(arr, axis = 0)
print arr.T
print np.transpose(arr)