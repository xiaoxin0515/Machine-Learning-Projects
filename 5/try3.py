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
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier


def combine_data(label):
    path = 'HMP_Dataset/' + label
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

def load_data(label):
    data_list = []
    path = 'HMP_Dataset/' + label
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

def segment(data, num, overlap):
    if overlap == 0:
        k = int(len(data)/num)
        end = num - 1 + k * num
    else:
        k = int((len(data) - num + 1) / (num * overlap))
        end = num - 1 + k * int(num * overlap)
    data = data[:end]
    index = 0
    # print(k)
    # print(end)
    for i in range(k):
        if k == 0:
            start = i * num
            end = num -1 + i * num
        else:
            start = i * int(num * overlap)
            end = num - 1 + i * int(num * overlap)
        # print(start)
        # print(end)
        single_segment = data[start:end+1]
        # print(single_segment)
        # print("#####################")
        single_segment = single_segment.reshape(1, -1)
        # print(single_segment)
        if index:
            data_segment = np.concatenate((data_segment, single_segment), axis=0)
        else:
            data_segment = single_segment
        index = 1

    return data_segment, k

def cluster_score(y_kmeans, cluster_num):
    num = len(y_kmeans)
    score = np.array([0 for i in range(cluster_num)])
    for y in y_kmeans:
        score[y] += 1
    percent = score/num

    return score




if __name__ == '__main__':
    # a = load_data('Brush_teeth')
    # print(len(a))
    # for i in a:
    #     print(i)
    #     print("##########")
    #label_list = ['brush_teeth', 'climb_stairs', 'comb_hair', 'descend_stairs', 'drink_glass', 'eat_meat', 'eat_soup', 'getup_bed', 'liedown_bed', 'pour_water', 'sitdown_chair', 'standup_chair', 'use_telephone', 'walk']
    label_list = ['brush_teeth', 'climb_stairs']
    train = True
    if not train:
        #######################################
        index = 0
        length = []  # length of
        for label in label_list:    # each activity
            data_list = load_data(label)
            # print(data_list)
            # print(len(data_list))
            length_list = []   # how many segments one signal have
            for i in range(len(data_list)):  # each signal
                #print(data_list[i].shape)
                data_segment, num = segment(data_list[i], 32, 0)
                # print(data_list[i])
                # print("###########")
                length_list.append(num)  # number of segments of each signal
                if index:
                    all_segment = np.concatenate((all_segment, data_segment))  # all segments of all activities
                else:
                    all_segment = data_segment
                index = 1
            length.append(length_list)

        data_output5 = open('variable1/length.pkl', 'wb')
        pickle.dump(length, data_output5)
        data_output5.close()
        #print(all_segment.shape)     # (6838, 96)
        kmeans = KMeans(n_clusters=480)
        kmeans.fit(all_segment)
        # centroid = kmeans.cluster_centers_
        # score = nearest_centroid(all_segment, centroid)
        # score = kmeans.fit_predict(all_segment)
        # print(score)
        # print(score.shape)       # (6838,)
        score = kmeans.predict(all_segment)
        data_output3 = open('variable1/all_segment.pkl', 'wb')
        pickle.dump(all_segment, data_output3)
        data_output3.close()
        data_output4 = open('variable1/score.pkl', 'wb')
        pickle.dump(score, data_output4)
        data_output4.close()

        # vector quantization
        sum = 0
        start = 0
        label_index = 0
        appearance_index = 0
        appearance_array = []
        label_result = []
        plot_index = 0
        appearance_by_label_array = []
        for label_length in length:  # how many files in one label
            appearance_by_label = []
            appearance_label_index = 0
            for segment_length in label_length:     # how many segments in one file
                # print(segment_length)
                segment_score = score[start:start+segment_length]
                # print(len(segment_score))
                start = segment_length
                # print(segment_score)
                # print("#############")
                segment_appearance, useless = np.histogram(segment_score, bins=np.arange(0, 481))
                # print(segment_appearance)
                # print(segment_appearance.shape)
                # print(segment_appearance)
                # print("##############")
                # print(segment_percent.shape)
                # appearance_array.append(segment_appearance)
                if appearance_index:
                    appearance_array = np.concatenate((appearance_array, segment_appearance))  # all the percent
                else:
                    appearance_array = segment_appearance
                appearance_index = 1
                label_result.append(label_list[label_index])
                # appearance_by_label.append(segment_score)
                if appearance_label_index:
                    appearance_by_label = np.concatenate((appearance_by_label, segment_score))
                else:
                    appearance_by_label = segment_score
                appearance_label_index = 1
            # label_index += 1
            # percent_by_label = percent_array.reshape(-1, 480)
            # hist_y = np.mean(percent_by_label, axis=0)
            # plt.subplot(5, 3, plot_index+1)
            # plt.hist(range(480), hist_y)
            # plt.title(label_list[label_index])
            # plot_index += 1


            # print(appearance_by_label)
            # appearance_by_label = np.array(appearance_by_label)
            # print(appearance_by_label)
            # print(appearance_by_label.shape)

            appearance_by_label_sum = np.sum(appearance_by_label, axis=0)
            appearance_by_label_mean = np.mean(appearance_by_label, axis=0)
            plt.subplot(5, 3, plot_index + 1)
            plt.hist(appearance_by_label, density=True, bins=480)
            plt.title(label_list[label_index])
            plot_index += 1

            label_index += 1
        plt.savefig('k=480')
        # appearance_array = np.array(appearance_array)
        appearance_array = appearance_array.reshape(-1, 480)
        # print(appearance_array.shape)
        # save variable as classifier input
        data_output1 = open('variable1/appearance_array.pkl', 'wb')
        pickle.dump(appearance_array, data_output1)
        data_output1.close()
        data_output2 = open('variable1/label_result.pkl', 'wb')
        pickle.dump(label_result, data_output2)
        data_output2.close()





    # np.savetxt('mean_percent_array', percent_array, delimiter=' ')
    # print(label_result)
    # # file = open('label_result.txt','w')
    # # for result in label_result:
    # #     file.write(result)
    # #     file.write('\n')
    # # file.close()
    # print(sum)
    # print(len(length))
    # print(length)
    #########################################
    else:
        # # load previous variable
        data_input1 = open('variable1/appearance_array.pkl', 'rb')
        data_all = pickle.load(data_input1)

        data_input1.close()
        data_input2 = open('variable1/label_result.pkl', 'rb')
        label_all = np.array(pickle.load(data_input2))
        data_input2.close()
        # print(data_all.shape)
        # print(len(label_all))

        # split data for cross validation, build classifier
        kf = KFold(n_splits=3)
        # model = RandomForestClassifier(n_estimators=100)
        # model.fit(data_all, label_all)
        # label_predict = model.predict(data_all)
        # accuracy_rate = accuracy_score(label_all, label_predict)
        # print(accuracy_rate)
        print(len(data_all))
        print(len(label_all))
        model = RandomForestClassifier(n_estimators=150, max_depth=120)

        # accuracy_rate = cross_val_score(model, data_all, label_all, cv=3)
        # print(accuracy_rate)

        # model = KNeighborsClassifier(n_neighbors=5)
        # accuracy_rate = cross_val_score(model, data_all, label_all, cv=3)
        # print(accuracy_rate)




        # print(len(data_all))
        # print(type(data_all))
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
        #     # print(ma)
        #     #print(label_predict)
        data_train = data_all[560:]
        data_test = data_all[0:560]
        label_train = label_all[560:]
        label_test = label_all[0:560]
        # model = RandomForestClassifier(n_estimators=150, max_depth=120)
        model = model.fit(data_train, label_train)
        label_predict = model.predict(data_test)
        accuracy_rate = accuracy_score(label_test, label_predict)
        print(accuracy_rate)
        print(model.score(data_test, label_test))





