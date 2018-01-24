
# -*- coding: UTF-8 -*-

import platform

version = platform.python_version()

class O(object):
    def __init__(self, *arg, **kwarg):
        print ('init')
        super().__init__(*arg, **kwarg)

    def __new__(cls, *arg, **kwarg):
        print ('new')
        return super().__new__(cls, *arg, **kwarg)

    def __call__(self, *arg, **kwarg):
        print ('call')


if __name__ == '__main__':
    if version >= '3.0.0':
        print 'true'
        oo = O()
        print ('_____')
        oo()
    else:
        print 'can not support these codes'