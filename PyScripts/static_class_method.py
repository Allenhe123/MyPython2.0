# -*- coding: UTF-8 -*-

class Foo(object):
    @staticmethod
    def staticfoo():
        print('call static method')

    @classmethod
    def classfoo(cls):
        print('call class method')


class MyTest:

    myname = 'peter'

    # add a instance attribute
    def __init__(self, name):
        self.name = name

    # class access class attribute
    def sayhello(self):
        print "say hello to %s" % MyTest.myname

    # instance can access class attribute
    def sayhello_1(self):
        print "say hello to %s" % self.myname

    # It's a snap! instance can access instance attribute
    def sayhello_2(self):
        print "say hello to %s" % self.name

    # class can not access instance attribute!!!
    def sayhello_3(self):
        #print "say hello to %s" % MyTest.name
        pass

if __name__ == '__main__':

    test = MyTest("abc")
    test.sayhello()
    test.sayhello_1()
    test.sayhello_2()
    test.sayhello_3()

    # class's definition can be changed dynamically
    MyTest.yourname = "Allen"

    print MyTest.myname
    print MyTest.yourname

############################################

    foo = Foo()
    foo.staticfoo()  # both instance and class can call static method
    Foo.staticfoo()  

    foo.classfoo()  # both instance and class can call class method
    Foo.classfoo()
