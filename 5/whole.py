from sklearn.cluster import KMeans
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

def combine_data(label_list):
    for label in label_list:
        path = 'HMP_Dataset 2/'
        files = os.listdir(path)
        # files = os.path.join(path, files)
        # print(files)
        index = 0
        for file in files:
            file_name = os.path.join(path, file)
            if index:
                data = np.concatenate((data, np.genfromtxt(file_name, delimiter=' ', skip_header=True)), axis=0)
            else:
                data = np.genfromtxt(file_name, delimiter=' ', skip_header=True)
            index = 1
    # data.reshape(1, data.shape[0] * data.shape[1])

    return data

def load_data(label):
    data_list = []
    path = 'HMP_Dataset 2/' + label
    files = os.listdir(path)
    files.sort()
    # num = len(files)
    for file in files:
        file_name = os.path.join(path, file)
        data = np.genfromtxt(file_name, delimiter=' ', skip_header=True)
        #data = data.reshape(1, data.shape[0] * data.shape[1])
        data_list.append(data)

    return data_list

# def combine_all_data(label_list):
#     index = 0
#     for label in label_list:
#         if index:
#             data_all = np.concatenate((data_all, combine_data(label)), axis=0)
#         else:
#             data_all = combine_data(label)
#         index = 1
#     np.savetxt('data/all_data', data_all, delimiter=' ')
#
#     return data_all

def nearest_centroid(data, cluster):
    nearest_lsit = []
    for segment in data:
        min_distance = float('inf')
        for i in range(len(cluster)):
            distance = np.linalg.norm(segment - cluster[i])
            if distance < min_distance:
                min_distance = distance
                nearest = i
        nearest_lsit.append(nearest)

    return nearest_lsit

def segment(data, num, overlap):
    if overlap == 0:
        k = int(len(data) - num/num)
        end = num + k * num
    else:
        k = int((len(data) - num) / int(num * overlap))
        end = num + k * int(num * overlap)
    data = data[:end]
    data_segment = []
    for i in range(k):
        if k == 0:
            start = i * num
            end = num + i * num
        else:
            start = i * int(num * overlap)
            end = num + i * int(num * overlap)
        single_segment = data[start:end]
        single_segment = single_segment.flatten()
        data_segment.append(single_segment)
        # single_segment = single_segment.reshape(1, -1)
        # print(single_segment.shape)
        # if index:
        #     data_segment = np.concatenate((data_segment, single_segment), axis=0)
        # else:
        #     data_segment = single_segment
        # index = 1

    # data_segment = data_segment.reshape(-1, 96)
    data_segment = np.array(data_segment)
    return data_segment, k

def cluster_score(y_kmeans, cluster_num):
    num = len(y_kmeans)
    score = np.array([0 for i in range(cluster_num)])
    for y in y_kmeans:
        score[y] += 1
    percent = score/num

    return score




if __name__ == '__main__':
    label_list = ['brush_teeth', 'climb_stairs', 'comb_hair', 'descend_stairs', 'drink_glass', 'eat_meat', 'eat_soup', 'getup_bed', 'liedown_bed', 'pour_water', 'sitdown_chair', 'standup_chair', 'use_telephone', 'walk']
    # label_list = ['brush_teeth', 'climb_stairs']
    train = True
    save = True
    save1 = True
    print("1")
    if not train:
        index = 0
        length = []  # length of
        for label in label_list:    # each activity
            data_list = load_data(label)
            length_list = []   # how many segments one signal have
            for i in range(len(data_list)):  # each signal
                data_segment, num = segment(data_list[i], 32, 0.5)
                length_list.append(num)  # number of segments of each signal
                if index:
                    all_segment = np.concatenate((all_segment, data_segment))  # all segments of all activities
                else:
                    all_segment = data_segment
                index = 1
            length.append(length_list)
        print("3")
        print(all_segment.shape)
        kmeans = KMeans(n_clusters=480)
        kmeans.fit(all_segment)
        # if save:
        #     data_output3 = open('new/kmeans.pkl', 'wb')
        #     pickle.dump(kmeans, data_output3)
        #     data_output3.close()







        score = kmeans.predict(all_segment)

        # # vector quantization
        # data_input3 = open('new/kmeans.pkl', 'rb')
        # kmeans = pickle.load(data_input3)
        # data_input3.close()

        #
        # label_result = []
        # appearance_by_label = []
        # index = 0
        # plot_index = 0
        # for label in label_list:    # each activity
        #     data_list = load_data(label)
        #     print("4")
        #     length_list = []   # how many segments one signal have
        #     for i in range(len(data_list)):  # each signal
        #         data_segment, num = segment(data_list[i], 32, 0.5)
        #         cluster = kmeans.predict(data_segment)
        #         appearance, useless = np.histogram(cluster, bins=np.arange(0, 481), density=True)
        #         label_result.append(label)
        #         appearance_by_label.append(appearance)
        #     appearance_by_label = np.array(appearance_by_label)
        #     plt.subplot(5, 3, plot_index + 1)
        #     plt.hist(appearance_by_label, density=True, bins=480)
        #     plt.title(label_list[label])
        #     plot_index += 1
        #
        #     if index:
        #         appearance_array = np.concatenate((appearance_array, appearance_by_label))
        #     else:
        #         appearance_array = appearance_by_label
        #     index += 1
        # appearance_array = appearance_array.reshape(-1, 480)
        # print(appearance_array.shape)
        # print(len(label_result))


        start = 0
        label_index = 0
        appearance_index = 0
        label_result = []
        plot_index = 0
        for label_length in length:  # how many files in one label
            appearance_by_label = 0
            appearance_label_index = 0
            for segment_length in label_length:     # how many segments in one file
                segment_score = score[start:start+segment_length]
                start = start + segment_length
                segment_appearance, useless = np.histogram(segment_score, bins=np.arange(0, 481), density=True)
                # a = cluster_score(segment_score, 480)
                # print(segment_appearance)
                if appearance_index:
                    appearance_array = np.concatenate((appearance_array, segment_appearance))  # all the percent
                else:
                    appearance_array = segment_appearance
                appearance_index = 1
                label_result.append(label_list[label_index])
                if appearance_label_index:
                    appearance_by_label = np.concatenate((appearance_by_label, segment_score))
                else:
                    appearance_by_label = segment_score
            # appearance_by_label_sum = np.sum(appearance_by_label, axis=0)
            # appearance_by_label_mean = np.mean(appearance_by_label, axis=0)
            plt.subplot(5, 3, plot_index + 1)
            plt.hist(appearance_by_label, density=True, bins=480)
            plt.title(label_list[label_index])
            plot_index += 1
            # plt.hist(appearance_by_label, density=False, bins=480)
            # plt.title(label_list[label_index])
            # plt.show()

            label_index += 1
        plt.savefig('k=480')
        appearance_array = np.array(appearance_array)
        appearance_array = appearance_array.reshape(-1, 480)
        print(appearance_array.shape)
        # save variable as classifier input
        if save1:
            data_output1 = open('new/appearance_array.pkl', 'wb')
            pickle.dump(appearance_array, data_output1)
            data_output1.close()
            data_output2 = open('new/label_result.pkl', 'wb')
            pickle.dump(label_result, data_output2)
            data_output2.close()

    else:
        # # load previous variable
        data_input1 = open('new/appearance_array.pkl', 'rb')
        data_all = pickle.load(data_input1)
        data_input1.close()
        data_input2 = open('new/label_result.pkl', 'rb')
        label_all = np.array(pickle.load(data_input2))
        data_input2.close()
        print(len(data_all))
        print(len(label_all))

        # split data for cross validation, build classifier


        ###3  try 1
        #
        model = RandomForestClassifier(n_estimators=240, max_depth=120)
        accuracy_rate = cross_val_score(model, data_all, label_all, cv=5)
        print(accuracy_rate)

        label_predict = cross_val_predict(model, data_all, label_all, cv=3)
        conf_mat = confusion_matrix(label_all, label_predict)
        print(conf_mat)


        #### try 2
        #
        # kf = KFold(n_splits=3)
        # for train_index, test_index in kf.split(data_all):
        #     data_train = data_all[train_index]
        #     data_test = data_all[test_index]
        #     label_train = label_all[train_index]
        #     label_test = label_all[test_index]
        #     #print(label_test)
        #     model = RandomForestClassifier(n_estimators=150, max_depth=120)
        #     model.fit(data_train, label_train)
        #     label_predict = model.predict(data_test)
        #     # for x in label_predict:
        #     #     for y in label_test:
        #     #         print(x == y)
        #     accuracy_rate = accuracy_score(label_test, label_predict)
        #     print(accuracy_rate)
        #     ma = confusion_matrix(label_test, label_predict, labels=label_list)
        #     print(ma)
        #     #print(label_predict)



        #### try 3
        # data_train = data_all[0:560]
        # data_test = data_all[560:]
        # label_train = label_all[0:560]
        # label_test = label_all[560:]
        # model = RandomForestClassifier(n_estimators=150, max_depth=120)
        # model = model.fit(data_train, label_train)
        # label_predict = model.predict(data_test)
        # accuracy_rate = accuracy_score(label_test, label_predict)
        # print(accuracy_rate)
        # print(model.score(data_test, label_test))

        # data_train = data_all[trainIndex]
        # print(data_train.shape)
        # data_test = data_all[testIndex]
        # print(data_test.shape)
        # label_train = label_all[trainIndex]
        # print(label_train.shape)
        # label_test = label_all[testIndex]
        # print(label_test.shape)
        # model = RandomForestClassifier(n_estimators=240, max_depth=120)
        # model.fit(data_train, label_train)
        # label_predict = model.predict(data_test)
        # # m= label_predict[:10]
        # # n = label_test[:10]
        # print(label_predict)
        # print(label_test)
        # print(np.count_nonzero(label_predict == label_test) * 1.0 / 279)
        # print(label_predict.shape)
        # # print(accuracy_rate(label_predict, label_test))
        # print(model.score(data_test, label_test))
