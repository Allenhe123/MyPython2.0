import socket
import time
import os


# def handle(client_sock, client_address)
#     while True:
#         data = client_sock.recv(4096).decode()
#         if data:
#             print("recv data: " + data)
#             sent = client_sock.send()

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(('127.0.0.1', 8888))
sock.listen(5)

while True:
    (clientsock, address) = sock.accept()
    data = clientsock.recv(4096).decode()
    print ('client name: ' + data)
    datatime = time.asctime() + '\n'
    msg = 'Hello ' + data
    clientsock.send(msg.encode())
    msg = 'My time is ' + datatime
    clientsock.send(msg.encode())
    clientsock.close()

sock.close()