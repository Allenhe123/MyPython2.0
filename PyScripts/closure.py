#!/bin/python
#-*-coding: utf-8 -*-

x = 100

'''
函数内部的作用域对于全局变量x仅有只读权限，想要在main()中对x进行改变，不会影响全局变量，而是会创建一个新的局部变量，
显然无法对还未创建的局部变量直接使用x += 1。如果想要获得全局变量的完全引用，则需要global声明
'''
'''
def func():
    x += 100
    print x
'''	

'''
要形成闭包，首先得有一个嵌套的函数，即函数中定义了另一个函数，闭包则是一个集合，它包括了外部函数的局部变量，
这些局部变量在外部函数返回后也继续存在，并能被内部函数引用。

'''

def func():
	global x
	x += 100
	print x

def func2():
	x = 99
	x += 100
	print x

'''
对于内部函数 nth_power，它能引用到外部函数的局部变量 n，而且即使 generate_power_func 已经返回。把这种现象就称为闭包
这就是闭包的作用，外部函数的局部变量可以被内部函数引用，即使外部函数已经返回了。

Python 中函数也是对象，所以函数也有很多属性，和闭包相关的就是 __closure__ 属性。__closure__属性定义的是一个包含 cell 
对象的元组，其中元组中的每一个 cell 对象用来保存作用域中变量的值。

所有函数都有一个 __closure__属性，如果这个函数是一个闭包的话，那么它返回的是一个由 cell 对象 组成的元组对象。
cell 对象的cell_contents 属性就是闭包中的自由变量。

在 power_func的 __closure__ 属性中有外部函数变量 n 的引用，通过内存地址可以发现，引用的都是同一个 n。如果没用形成闭包，
则 __closure__ 属性为 None。

闭包函数都有一个__closure__属性，其中包含了它所引用的上层作用域中的变量

一般来说，当对象中只有一个方法时，这时使用闭包是更好的选择。这比用类来实现更优雅，此外装饰器也是基于闭包的一中应用场景。

闭包避免了使用全局变量，此外，闭包允许将函数与其所操作的某些数据（环境）关连起来。这一点与面向对象编程是非常类似的，
在面对象编程中，对象允许我们将某些数据（对象的属性）与一个或者多个方法相关联。

'''
def generate_power_func(n):
    def nth_power(x):
        return x ** n
    return nth_power

def adder(x):
    def wrapper(y):
        return x + y
    return wrapper

def foo():
    m = 100
    def goo(x):
        return m + x
    return goo

if __name__ == '__main__':
    print x  #100
    func()   #200
    print x  #200
    func2()  #199

    power_func = generate_power_func(2)
    print power_func(4)   #16

    print power_func.__closure__  #(<cell at 0x7f92f1dab6e0: int object at 0xfe9140>,)
    print type(power_func.__closure__[0]) #<type 'cell'>
    print power_func.__closure__[0].cell_contents  #2

    adder5 = adder(5)
    print adder5(6)     #11

    fooer = foo()
    print fooer(8)    #108
