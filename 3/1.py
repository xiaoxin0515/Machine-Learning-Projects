import numpy as np
from numpy import genfromtxt
from sklearn.decomposition import PCA

def pca(data, k):
    row, column = np.shape(data)
    average = np.mean(data, axis=0)
    trans_mean = data - np.tile(average, (row, 1))
    covmat = np.cov(trans_mean.T)
    eigValue, eigVec = np.linalg.eig(covmat)
    eigValue_descent = np.argsort(-eigValue)
    selectVec = np.matrix(eigVec.T[:k])
    selectVec = selectVec.T
    #print(np.shape(trans_mean))

    return selectVec

def inverse(trans_mean, u):
    new = np.dot(u.T, trans_mean.T)
    new = np.dot(u, new)
    new = new.T

    return new

def error(data1, data2):
    error = 0
    data = data1 - data2
    #print(data)
    row, column = np.shape(data)
    for i in range(row):
        for j in range(column):
            error += data[i][j] ** 2
    mse = error/row

    return mse

def zero_part(data_compare, data_mean):
    row, column = np.shape(data_compare)
    average = np.mean(data_mean, axis=0)
    data_new = np.tile(average, (row, 1))
    mse = error(data_new, data_compare)
    print(mse)
    #print("the mse of 0 is:" + " " + str(mse))

    return mse

def N_part(data, name, data2):
    row, column = np.shape(data)
    mse_list = []
    print(name)
    mse0 = zero_part(data2, data2)
    mse_list.append(mse0)
    for component_num in range(1, 5):
        u = pca(data2, component_num)
        #print(u)
        average = np.mean(data2, axis=0)
        trans_mean = data - np.tile(average, (row, 1))
        data_new = inverse(trans_mean, u) + average
        data_new = np.array(data_new)
        mse = error(data_new, data_noiseless)
        #print(mse)
        mse_list.append(mse)
        print("the mse of " + str(component_num) + "N is:" + " " + str(mse))

    return mse_list

def c_part1(data, name, data_compare, mse_list, save=False):
    print(name)
    mse0 = zero_part(data_compare, data)
    mse_list.append(mse0)
    # when 0c:
    for component_num in range(1, 5):
        pca_function = PCA(n_components=component_num)
        data_new = pca_function.fit_transform(data)
        data_inverse = pca_function.inverse_transform(data_new)
        if (save):
            if (component_num == 2):
                np.savetxt('xiaoxin2-recon.csv', data_inverse, delimiter=',')
        #print(data_inverse)
        mse = error(data_compare, data_inverse)
        mse_list.append(mse)
        print("the mse of " + str(component_num) + "c is:" + " " + str(mse))

    return mse_list


data_noiseless = genfromtxt("iris.csv", delimiter=",", skip_header=True)
data1 = genfromtxt("dataI.csv", delimiter=",", skip_header=True)
data2 = genfromtxt("dataII.csv", delimiter=",", skip_header=True)
data3 = genfromtxt("dataIII.csv", delimiter=",", skip_header=True)
data4 = genfromtxt("dataIV.csv", delimiter=",", skip_header=True)
data5 = genfromtxt("dataV.csv", delimiter=",", skip_header=True)

zero_part(data_noiseless, data_noiseless)
mse1 = N_part(data1, "data1", data_noiseless)
c_part1(data1, "data1", data_noiseless, mse1, save=True)
mse2 = N_part(data2, "data2", data_noiseless)
c_part1(data2, "data2", data_noiseless, mse2)
mse3 = N_part(data3, "data3", data_noiseless)
c_part1(data3, "data3", data_noiseless, mse3)
mse4 = N_part(data4, "data4", data_noiseless)
c_part1(data4, "data4", data_noiseless, mse4)
mse5 = N_part(data5, "data5", data_noiseless)
c_part1(data5, "data5", data_noiseless, mse5)

print(mse1)
print(mse2)
print(mse3)
print(mse4)
print(mse5)







