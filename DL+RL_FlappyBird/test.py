import numpy as np
from collections import deque

a = np.array([[1,2,3],[4,5,6],[7,8,9]])
#print a
a1 = np.array([[11,12,13],[14,15,16],[17,18,19]])
a2 = np.array([[21,22,23],[24,25,26],[27,28,29]])
a3 = np.array([[31,32,33],[34,35,36],[37,38,39]])
a4 = np.array([[41,42,43],[44,45,46],[47,48,49]])
b = np.stack((a,a1,a2,a3))
#print b

c = np.reshape(b, (3,3,4))
#print c

d = np.stack((a,a1,a2,a3),axis=2)
print d
print ("\n")

e = d[:,:,1:]
print e

g = np.reshape(a4, [3,3,1])
print g
f = np.append(e, g,axis = 2)
print f

reward = 1

ter = 0

D = deque()

D.append((f,reward,ter))
print D

D.append((d,0,1))
print D
D.popleft()
print D