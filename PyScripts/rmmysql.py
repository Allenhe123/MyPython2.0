#!/usr/bin/python

import os

if __name__ == '__main__':
	v = os.system('ls')
	print "v = ", v
	
	result = os.popen('ls').read()
	print result	

	ret = os.popen('ls | wc -l').read()
	print ret

	os.system('echo "uninstall mysql..."')
	os.system('rm -R /var/lib/mysql/')
	os.system('rm -R /etc/mysql/')
	os.system('apt-get autoremove mysql* --purge')
	os.system('apt-get remove apparmor')
#	os.system('apt-get install mysql-server mysql-common')

'''
do not concern return result of shell command use os.system
otherwise use os.popen
'''
