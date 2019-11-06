import numpy as np
import random
from sympy import *
import pandas as pd
import matplotlib.pyplot as plt
import math
#
# x = range(10)
# y1 = [elem*2 for elem in x]
# plt.plot(x, y1)
#
# y2 = [elem**2 for elem in x]
# plt.plot(x, y2, 'r--')
#
# plt.show()


# x = [0,1,2,3,4,5,6,7,8,9]
# y1 = [elem*2 for elem in x]
# plt.plot(x, y1)
#
# y2 = [elem**2 for elem in x]
# plt.plot(x, y2, 'r--')
#
# plt.show()



# a=np.arange(5).reshape(1,5)
# b=np.random.normal(0,1,4)
# c=b.T
# print(a.size)


# a=np.arange(5).reshape(1,5)
# b=1
# x=symbols("x")
# y=

# index_validation = random.sample(list(range(43957)),43957//10)
# print(len(index_validation))
# index_train = list(set(list(range(43957)))-set(index_validation))
# print(len(index_train))
# print(4395+39562)

#
# train = pd.read_csv("train.txt",header = None)
# index_update = random.sample(list(range(len(train))),1)
# data_update = train.iloc[index_update]
# series_random = list(data_update.iloc[:,:7])
# x = series_random[:6]
# y = series_random[6]
# print(x)
# print(y)
# for i in train:
#     print(i)
#     print("##########")
#
# a=[1,2,3]
# b=np.array(a)
# print(a)
# print(b)

# a = np.random.normal(0,1,6)
# b = np.random.normal(0,1,1)
#
# def magnitude(a,b):
#     mag = 0
#     for i in a:
#         mag += i**2
#     mag += b**2
#     return math.sqrt(mag)
#
# x=magnitude(a,b)
# np.linalg.norm(x)

a=[1,2,3]
b=[1,2,2]
c=[100,111,112]
d=[243,13,329]
fig1=plt.figure("fig1")
plt.plot(a,b)
fig2=plt.figure("fig2")
plt.plot(c,d)

fig1.savefig("1")
fig2.savefig("2")
