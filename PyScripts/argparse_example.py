# -*- coding: UTF-8 -*-

import argparse

# python argparse_example.py --help
# python argparse_example.py --h 可以打印出参数说明和帮助信息

#python argparse_example.py 'hello world' 3 --verbosity=0 -v -o --fuck=1 

#位置参数必须指定
#可选参数可以不加，不加就为None。一旦加了如果action指定了有默认值的就不赋值，没有默认值的必须用=号赋值
# 默认可选参数会被赋值为str，而在实际情况中，很多时候，可选参数是作为一个标识而不是一个确切的值，
# 仅需要确定是true或false即可，可以指定关键字action，赋值为"store_true".

# 注意，参数顺序没有关系

parse = argparse.ArgumentParser(description="information")

# 位置参数---指定位置参数后，程序必须输入该参数值才能运行
parse.add_argument("echo", help='cho the string you use here')
parse.add_argument("square", help="display a square of a given number", default=2, type = int)

# 可选参数---设置可选参数后，程序可以不输入该参数值运行.
# 如果程序运行时要赋值可选参数，必须先指定该可选参数。
parse.add_argument("--verbosity", help="increase output verbosity") 

# --verbose和--open可以像--help类似，只需输入参数即可
parse.add_argument("-v", "--verbose", help="add action for verbose", action='store_true') 

# 快捷选项（短选项） -o是--open的短选项
parse.add_argument("-o", "--open", help="o is short verison of open", action="store_true")

# 指定关键字default来指定选项默认的值. 指定可选参数的取值范围，用关键字choices
parse.add_argument("--fuck", help="add fuck", default=2, type=int, choices=[0,1,2])



args = parse.parse_args()

print 'args: ', args
print 'args.echo: ', args.echo
print 'args.square: ', args.square
answer = args.square * 2
print 'answer is: ', answer
if args.verbosity:  
    print "verbosity turned on"
if args.verbose:
    print 'verbose added'
    print "the square of {} equals {}".format(args.square, answer)
if args.open:
    print 'open added'
if args.fuck == 1:
    print 'fuck= 1'

#可以使用可选参数来显示不同级别的信息
#{}对应小数点后的参数
if args.verbosity >= 2:
    print "Running '{}'".format(__file__)
if args.verbosity >= 1:
    print "{}^{}==".format(args.echo, args.square)

