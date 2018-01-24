# -*- coding: UTF-8 -*-

def div(a, b):
    try:
        print(a / b)
    except ZeroDivisionError:
        print('Error: b should not be 0!!')
    except Exception as e:
        print('Unexceptd Error: {}'.format(e))
    else:
        print('Run into else only when everything goes well')
    finally:
        print('Always run into finally block.')

def div1(a, b):
    # Mutiple exception in one line
    try:
        print(a / b)
    except (ZeroDivisionError, TypeError) as e:
    # except (ZeroDivisionError, TypeError):
    #     print('ZZZZZ')
        print(e)

def open_database():
    # Except block is optional when there is finally
    try:
        open(database)
    finally:
        close(database)

def do_work():
    # catch all errors and log it
    try:
        do_work()
    except:    
    # get detail from logging module
        logging.exception('Exception caught!')
    
    # get detail from sys.exc_info() method
    error_type, error_value, trace_back = sys.exc_info()
    print(error_value)
    raise

def open_file(filename):
    try:
        f = open(filename, 'r')
        print('open successful...')
        raise
    except Exception as e:
        print('Unexceptd Error: {}'.format(e))
    else:
        print('if no exception raised, come here')
    finally:
        print('before close file test.py')
        f.close()
        print('after close file test.py')

def open_file_with(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        for l in lines:
            print(l)

def assert_(expr, args=none):
    if __debug__ and not expr:
        raise AssertionError, args

if __name__ == '__main__':
    div(2, 0)
    div(2, 'bad_type')
    div(1, 2)
    div1(1, 0)
    open_file('test.py')
    open_file_with('super_call_.py')




# except语句不是必须的，finally语句也不是必须的，但是二者必须要有一个，否则就没有try的意义了。
# except语句可以有多个，Python会按except语句的顺序依次匹配你指定的异常，如果异常已经处理就不会再进入后面的except语句。
# except语句可以以元组形式同时指定多个异常，参见实例代码。
# except语句后面如果不指定异常类型，则默认捕获所有异常，你可以通过logging或者sys模块获取当前异常。
# 如果要捕获异常后要重复抛出，请使用raise，后面不要带任何参数或信息。
# 不建议捕获并抛出同一个异常，请考虑重构你的代码。
# 不建议在不清楚逻辑的情况下捕获所有异常，有可能你隐藏了很严重的问题。
# 尽量使用内置的异常处理语句来 替换try/except语句，比如with语句，getattr()方法。

 
