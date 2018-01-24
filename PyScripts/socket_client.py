
import socket
import os


def GetUsername():
    '''Attempt to find the username in a cross-platform fashion.'''
    try:
        return os.getenv('USER') or os.getenv('LOGNAME') or \
            os.getenv('USERNAME') or \
            os.getlogin() or 'nobody'
    except (AttributeError, IOError, OSError):
        return 'nobody'

user = GetUsername()
#print (user)

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('127.0.0.1', 8888))
sock.send(user.encode())
#sock.send(os.getlogin() + '\n')
message = sock.recv(1024).decode()
print (message)
sock.close()