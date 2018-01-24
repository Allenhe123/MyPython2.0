# -*- coding: UTF-8 -*-

import sys
import getopt

print "script name: ", sys.argv[0]
print 'param number: ', len(sys.argv) - 1
for i in range(1, len(sys.argv)):
    print "param ", i, sys.argv[i]

#################
# options返回元组的列表，args返回列表
# 输入：命令行参数列表，'短选项'，‘长选项列表’
# 短选项名后的冒号(:)表示该选项必须有附加的参数，不带冒号表示该选项不附加参数。
# 长选项名后的等号(=)表示如果设置该选项，必须有附加的参数，否则就不附加参数。

# python getopt_example.py -a -b -c ccc -o ooo -v --output=output --verbose --version=version --file file param1 param2 param3
try:
    options, args = getopt.getopt(sys.argv[1:], 'abc:o:v', ['output=', 'verbose', 'version=', 'file='])
    print 'options: ', options
    print 'args: ', args
except getopt.GetoptError as err:
    print 'ERROR:', err
    sys.exit(1)