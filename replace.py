import os, re

f = open("/home/user/seasons.txt", 'r')
f_other = open("/home/user/season.txt", 'w')

l = f.readline()
while l:
	rep_line = l.replace("-", "_")
	f_other.write(rep_line)
	l = f.readline()

f.close()
f_other.close()	