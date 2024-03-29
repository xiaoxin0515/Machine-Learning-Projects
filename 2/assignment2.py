import random
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def split(train):
    index_envaluation = random.sample(list(range(len(train))), 50)
    envaluation = train.iloc[index_envaluation]
    index_train = list(set(list(range(len(train)))) - set(index_envaluation))
    train = train.iloc[index_train]

    return train, envaluation

def generate_u():
    a = np.random.normal(0, 1, 6)
    b = np.random.normal(0, 1, 1)
    return a,b

def cost_function(data,a,b,lam):
    num = data.shape[0]
    penalty = 0
    err_cost_sum = 0
    for i in range(num):
        label_predict = 0
        series = list(data.iloc[i,:7])   # when use iloc to slice from... to...(lenth>1), it's just like range(use :6 to get 0-5)
                              # but when use it to get a exact location, use 6 is to just get 6th column(start at 0)
        x = series[:6]
        y = series[6]
        for j in range(6):
            label_predict += x[j] * a[j]
        label_predict += b
        sign = 1 - y * label_predict
        if (sign >= 0):
            err_cost = 0
        else:
            err_cost = sign
        err_cost_sum += err_cost
    for i in range(6):
        penalty += a[i] ** 2
    penalty = penalty * lam
    cost = err_cost_sum / num + penalty

    return cost

def SGD(data, a, b, eta, lam):
    num = data.shape[0]
    a_new = a
    p_a = [0,0,0,0,0,0]
    p_b = 0
    for i in range(num):
        label_predict = 0
        series = list(data.iloc[i, :7])  # when use iloc to slice from... to...(lenth>1), it's just like range(use :6 to get 0-5)
        # but when use it to get a exact location, use 6 is to just get 6th column(start at 0)
        x = series[:6]
        y = series[6]
        for j in range(6):
            label_predict += x[j] * a[j]
        label_predict += b
        sign = 1 - y * label_predict
        if (sign < 0):
            p_b += -y
            for k in range(6):
                p_a[k] += -y * x[k]
    for i in range(6):
        a_new[i] = a[i] - eta * (p_a[i]/num + lam*a[i])
    b_new = b - eta * p_b/num

    return a_new, b_new
# def SGD(data, a, b, eta, lam):
#     a_new = a
#     series = list(data.iloc[:, :7])
#     x = series[:6]
#     y = series[6]
#     label_predict = 0
#     for i in range(6):
#         label_predict += x[i] * a[i]
#     label_predict += b
#     sign = 1 - y * label_predict
#     if(sign>=0):
#         b_new = b
#         #b = b - eta * 0
#         for i in range(6):
#             a_new[i] = a[i] - eta * lam *a[i]
#     else:
#         b_new = b + eta * y
#         for i in range(6):
#             a_new[i] = a[i] - eta * (lam * a[i] - y * x[i])
#
#     return a_new, b_new

def accuracy(data, a, b):
    num = data.shape[0]
    acc_num = 0
    for i in range(num):
        label_predict = 0
        series = list(data.iloc[i, :7])  # when use iloc to slice from... to...(lenth>1), it's just like range(use :6 to get 0-5)
        # but when use it to get a exact location, use 6 is to just get 6th column(start at 0)
        x = series[:6]
        y = series[6]
        for j in range(6):
            label_predict += x[j] * a[j]
        label_predict += b
        if(y * label_predict >= 0):
            acc_num += 1
    acc_rate = acc_num/num

    return acc_rate

def magnitude(a):
    mag = 0
    for i in a:
        mag += i**2
    return math.sqrt(mag)

def predict(x, a, b):
    label_predict = 0
    for i in range(6):
        label_predict += a[i] * x[i]
    label_predict += b
    #print(label_predict)
    if(label_predict >= 0):
        label = " >50K"
    else:
        label = " <=50K"
    return label


if __name__ == '__main__':
    data_train = pd.read_csv("train.txt", header=None)
    data_train = data_train.iloc[:, [0, 2, 4, 10, 11, 12, 14]]
    # train.iloc[:, 6] = list(map(lambda x: 1 if x == " >50K" else -1, train.iloc[:, 6]))
    data_train.iloc[:, 6] = data_train.iloc[:, 6].apply(lambda x: 1 if x == " >50K" else -1)
    for i in range(6):
        data_train.iloc[:, i] = (data_train.iloc[:, i] - np.mean(data_train.iloc[:, i])) / np.std(data_train.iloc[:, i])
    index_validation = random.sample(list(range(43957)), 43957 // 10)
    validation = data_train.iloc[index_validation]
    index_train = list(set(list(range(43957))) - set(index_validation))
    data_train = data_train.iloc[index_train]


    data_test = pd.read_csv("test.txt", header=None)
    data_test = data_test.iloc[:, [0, 2, 4, 10, 11, 12]]
    for i in range(6):
        data_test.iloc[:, i] = (data_test.iloc[:, i] - np.mean(data_test.iloc[:, i])) / np.std(data_test.iloc[:, i])


    lam_range = [1e-3, 1e-2, 1e-1, 1]
    color = ["red", "orange", "yellow", "green"]
    eta = 0.02
    # a, b = generate_u()   # same initialization for 4 different regularization constants


# start training
    for i in range(4):      # different regularization constant (lam)
        a, b = generate_u()  # same initialization for 4 different regularization constants
        lam = lam_range[i]
        train, evaluation = split(data_train)
        k = 1  # step
        plot_x = []
        plot_acc_y = []#     return a_new, b_new
        plot_cost_y = []
        plot_magnitude_y = []
        for season in range(50): #50
            print("########  season  " + str(season + 1) + "########")
            #eta = 1/(season * 0.01 + 50)
            for step in range(300): #300
                if(k%30 == 0):
                    accuracy_rate = accuracy(evaluation, a, b)
                    mag = magnitude(a)
                    plot_x.append(k)
                    plot_acc_y.append(accuracy_rate)
                    plot_magnitude_y.append(mag)

                index_update = random.sample(list(range(len(train))), 1)
                data_update = train.iloc[index_update]
                a, b = SGD(data_update, a, b, eta, lam)
                k += 1


        fig1 = plt.figure("fig1")
        plt.title("validation accuracy every 30 steps of 4 regularization constant")
        plt.plot(plot_x, plot_acc_y, color=color[i], label="lam="+str(lam))
        plt.xlabel("step")
        plt.ylabel("accuracy rate")
        plt.legend()


        fig2 = plt.figure("fig2")
        plt.title("magnitude of coefficient vector every 30 steps of 4 regularization constant")
        plt.plot(plot_x, plot_magnitude_y, color=color[i], label="lam="+str(lam))
        plt.xlabel("step")
        plt.ylabel("magnitude of coefficient vector")
        plt.legend()

        accuracy_final = accuracy(validation, a, b)
        print("accuracy rate when regularization constant = " + str(lam) + " is " + str(accuracy_final))

        # predict the label of test dataset
        result = []
        num_test = data_test.shape[0]
        for i in range(num_test):
            series = list(data_test.iloc[i, :6])
            x = series[:6]
            y = predict(x, a, b)
            result.append(y)
        file = open('final/predict.txt', 'a')
        file.write("*************************")
        file.write("lam=" + str(lam) + "b = " + str(b) + "a = ")
        for a_single in a:
            file.write(str(a_single))
            file.write("  ")
        file.write("\n")
        for y in result:
            file.write(y)
            file.write('\n')
        file.close()

    fig1.savefig("final/result1")
    fig2.savefig("final/result2")
    # fig1.show()
    # fig2.show()
    # plt.show()









