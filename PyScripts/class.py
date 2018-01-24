# -*- coding: UTF-8 -*-

class Foo(object):
    refCount = 0

    def __init__(self):
        Foo.refCount += 1

    def __del__(self):
        Foo.refCount -= 1

    def howmanyRef(self):
        return Foo.refCount

if __name__ == '__main__':
    print(dir(Foo))   #
    print('\n')
    print(Foo.__dict__)
    print('\n')
    print(vars(Foo))  #与Foo.__dict___等价
    print('\n')
    print(type(Foo))
    print('\n')
    print(Foo.__name__)
    print('\n')
    print(Foo.__module__)
    print('\n')
    foo = Foo()
    print(foo.howmanyRef())

    print(Foo.refCount)
    print(foo.refCount)
    foo.refCount += 100
    print(foo.refCount)
    # Foo.refCount += 100
    print(Foo.refCount)

# dir() 是 Python 提供的一个 API 函数，dir() 函数会自动寻找一个对象的所有属性，
# 包括搜索 __dict__ 中列出的属性。
# 不是所有的对象都有 __dict__ 属性。例如，如果你在一个类中添加了 __slots__ 属性，
# 那么这个类的实例将不会拥有 __dict__ 属性，但是 dir() 
# 仍然可以找到并列出它的实例所有有效属性。

# 同理许多内建类型都没有 __dict__ 属性，例如 list 没有 __dict__ 属性，
# 但你仍然可以通过 dir() 列出 list 所有的属性。

# 一个实例的 __dict__ 属性仅仅是那个实例的局部属性集合，不包含该实例所有有效属性。
# 所以如果你想获取一个对象所有有效属性，你应该使用 dir() 来替代 __dict__ 或者 __slots__。

