import numpy as np
from numpy import linalg as la
from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

a = np.array([[ 1.  ,  1.  ],
       [ 0.9 ,  0.95],
       [ 1.01,  1.03],
       [ 2.  ,  2.  ],
       [ 2.03,  2.06],
       [ 1.98,  1.89],
       [ 3.  ,  3.  ],
       [ 3.03,  3.05],
       [ 2.89,  3.1 ],
       [ 4.  ,  4.  ],
       [ 4.06,  4.02],
       [ 3.97,  4.01]])
pca=PCA(n_components=1)
newData=pca.fit_transform(a)
b=pca.inverse_transform(newData)
print(b)

# def pca(dataMat, k):
#     average = np.mean(dataMat, axis=0) #按列求均值
#     m, n = np.shape(dataMat)
#     meanRemoved = dataMat - np.tile(average, (m,1)) #减去均值
#     normData = meanRemoved / np.std(dataMat) #标准差归一化
#     covMat = np.cov(normData.T)  #求协方差矩阵
#     eigValue, eigVec = np.linalg.eig(covMat) #求协方差矩阵的特征值和特征向量
#     eigValInd = np.argsort(-eigValue) #返回特征值由大到小排序的下标
#     selectVec = np.matrix(eigVec.T[:k]) #因为[:k]表示前k行，因此之前需要转置处理（选择前k个大的特征值）
#     finalData = normData * selectVec.T #再转置回来
#     return finalData