import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
import pickle
from sklearn.feature_extraction import stop_words
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

def accuracy_rate(model, threshold, feature, label):
    prob = model.predict_proba(feature)
    prob_pos = prob[:, 1]

    predict = []
    for p in prob_pos:
        if p > threshold:
            pre = 1
        else:
            pre = 0
        predict.append(pre)
    predict = np.array(predict)
    correct = np.count_nonzero(predict == label)
    rate = correct / len(label)

    return rate

def word_frequency(max_frequency, stop_file, modify=False):
    cv = CountVectorizer(stop_words=stop_file, lowercase=True, max_df=max_frequency)
    cv_fit = cv.fit_transform(text)
    count = cv_fit.toarray()
    count_sum = count.sum(axis=0)
    count_ordered = sorted(count_sum, reverse=True)
    word = cv.get_feature_names()
    word_dict = dict(zip(word, count_sum))
    count_dict_sorted = sorted(word_dict.items(), key=lambda item: item[1])

    data_output1 = open('count_ordered.pkl','wb')
    pickle.dump(count_ordered, data_output1)
    data_output1.close()

    plt.scatter(range(len(count_ordered)), count_ordered)
    plt.title('Word Frequency')
    plt.xlabel('Word Rank')
    plt.ylabel('Word Count')
    if modify:
        plt.savefig('word_frequency_modify')
        plt.show()
    else:
        plt.savefig('word_frequency')

    if modify == False:
        return word, cv_fit, count_dict_sorted
    else:
        print(len(count_ordered))
        return 0


if __name__ == '__main__':
    data_whole = pd.read_csv('yelp_2k.csv')
    data = data_whole.loc[:, ['stars', 'text']]
    data['label'] = data.stars.apply(lambda x: 0 if x==1 else 1)
    text = data.text
    label = data.label


###### first plot ##########
    cv = CountVectorizer(stop_words=None, lowercase=True)
    cv_fit = cv.fit_transform(text)
    count = cv_fit.toarray()
    count_sum = count.sum(axis=0)
    count_ordered = sorted(count_sum, reverse=True)
    word = cv.get_feature_names()
    word_dict = dict(zip(word, count_sum))
    count_dict_sorted = sorted(word_dict.items(), key=lambda item: item[1])

    # data_output1 = open('count_ordered.pkl', 'wb')
    # pickle.dump(count_ordered, data_output1)
    # data_output1.close()

    plt.scatter(range(len(count_ordered)), count_ordered)
    plt.title('Word Frequency')
    plt.xlabel('Word Rank')
    plt.ylabel('Word Count')
    plt.savefig('word_frequency')
    plt.close()


###### prune ##########
    tf = TfidfTransformer()
    tfidf = tf.fit_transform(cv_fit).toarray()
    tfidf_sum = tfidf.sum(axis=0)
    tfidf_dict = dict(zip(word, tfidf_sum))
    tfidf_sorted = sorted(tfidf_dict.items(), key=lambda item: item[1], reverse=True)
    stop_word = tfidf_sorted[0:10]
    stop_list = []
    for t in stop_word:
        stop_list.append(t[0])

    max_count_stop = 4000
    stop_list.extend(stop_words.ENGLISH_STOP_WORDS)
    for i in range(len(count_dict_sorted) - 1, -1, -1):
        # print(i)
        c = count_dict_sorted[i]
        # print(c)
        if c[1] > max_count_stop:
            stop_list.append(c[0])
        else:
            break
    data_output1 = open('stop_list.pkl', 'wb')
    pickle.dump(count_ordered, data_output1)
    data_output1.close()
    print(len(stop_list))

    min_list = []
    min_count = 10
    for c in count_dict_sorted:
        if c[1] < min_count:
            min_list.append(c[0])
        else:
            break
    print(len(min_list))
    stop_list.extend(min_list)


######## modified plot ############
    cv_modify = CountVectorizer(stop_words=stop_list, lowercase=True, max_df=5000)
    cv_fit_modify = cv_modify.fit_transform(text)
    count_modify = cv_fit_modify.toarray()
    count_sum_modify = count_modify.sum(axis=0)
    count_ordered_modify = sorted(count_sum_modify, reverse=True)
    vocabulary = cv_modify.get_feature_names()
    print(len(vocabulary))

    plt.scatter(range(len(count_ordered_modify)), count_ordered_modify)
    plt.title('Word Frequency')
    plt.xlabel('Word Rank')
    plt.ylabel('Word Count')
    plt.savefig('word_frequency_modify')
    plt.close()

    feature = count_modify
    # print(feature)

################# horrible service ##############
    # sentence = ['Horrible customer service', ]
    # cv_sentence = CountVectorizer(lowercase=True, vocabulary=vocabulary)
    # cv_fit_sentence = cv_sentence.fit_transform(sentence)
    # vector_sentence = cv_fit_sentence.toarray()
    # print(vector_sentence)
    #
    # #
    # # dist = []
    # # for i in count_modify:
    # #     # print(i)
    # #     # print(vector_sentence)
    # #     # vector = i.reshape(1, -1)
    # #     # print(vector.shape)
    # #     # print(vector_sentence)
    # #     d = result1 = 1 - spatial.distance.cosine(i, vector_sentence)
    # #     dist.append(d)
    # # # data_output2 = open('distance.pkl', 'wb')
    # # # pickle.dump(distance, data_output2)
    # # # data_output2.close()
    #
    #
    # # print(dist)
    # similarity = cosine_similarity(vector_sentence, count_modify)
    # index_sort = np.argsort(-similarity)
    # index_sort = index_sort[0]
    # index = index_sort[0:5]
    # # index = index[0]
    # print(index)
    # #
    # # with open('text_horrible', 'a') as file:
    # #     for i in index:
    # #         # print(i)
    # #         file.write(text.iloc[i])
    # #         file.write('\n')
    # #         file.write("########################")
    # #         file.write('\n')
    # #         # print(type(text.iloc[i]))
    # # file.close()
    # # plt.scatter(range(len(similarity_sort)), similarity_sort)
    # # plt.show()
    # similarity_sort = np.sort(similarity)
    # similarity_sort = similarity_sort[0]
    # plt.scatter(range(len(similarity_sort)), similarity_sort)
    # plt.show()
    # print(len(similarity_sort[similarity_sort > 0.4]))
    #

################### part 3 ###################
    random.seed(0)
    index_test = random.sample(list(range(len(feature))), 200)
    index_train = list(set(list(range(len(feature)))) - set(index_test))
    train = feature[index_train, :]
    test = feature[index_test, :]
    train_label = np.array(label.iloc[index_train])
    test_label = np.array(label.iloc[index_test])

    lr = LogisticRegression(solver='lbfgs')
    lr.fit(train, train_label)
    predict1 = lr.predict(test)
    print(lr.score(test, test_label))
    print(lr.score(train, train_label))

######### plot ############
    # prob_all = lr.predict_proba(feature)
    # index_pos = np.where(label == 1)
    # pos = prob_all[index_pos]
    #
    # index_neg = np.where(label == 0)
    # neg = prob_all[index_neg]
    #
    # fig1 = plt.figure('prob')
    # plt.title('histogram of predicted scores')
    # plt.xlabel('predicted score')
    # plt.ylabel('count of predictions in bucket')
    # plt.hist(pos[:, 1], color='green', bins=100)
    # plt.hist(neg[:, 1], color='blue', bins=100)
    # plt.savefig('prob')
    # plt.show()

######### change threshold ###########
    threshold = 0.6

    rate_train = accuracy_rate(lr, threshold, train, train_label)
    rate_test = accuracy_rate(lr, threshold, test, test_label)

    print(rate_train)
    print(rate_test)



    # print(test_label)
    fpr, tpr, thresholds = roc_curve(test_label, lr.decision_function(test))
    # print(len(threshold))
    # print(type(fpr))
    # print(fpr)
    # print(fpr.shape)
    # point = np.concatenate(fpr, tpr, axis=1)
    # print(point)
    roc_auc = auc(fpr, tpr)

    # plt.figure()
    # plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    # plt.legend(loc="lower right")
    # plt.tile('ROC curve')
    # plt.xlabel('false positive rate')
    # plt.ylabel('true positive rate')
    # plt.show()



    # min_dist = float('inf')
    # for i in range(len(fpr)):
    #     dist = np.sqrt(np.square(fpr[i]-0) + np.square(tpr[i]-1))
    #     if dist < min_dist:
    #         min_dist = dist
    #         index = i
    #         fpr_min = fpr[i]
    #         tpr_min = tpr[i]
    # print(min_dist)
    # print(fpr_min)
    # print(tpr_min)
    # print(index)
    # print(thresholds[index])

    # i = np.arange(len(tpr))  # index for df
    # roc = pd.DataFrame(
    #     {'fpr': pd.Series(fpr, index=i), 'tpr': pd.Series(tpr, index=i), '1-fpr': pd.Series(1 - fpr, index=i),
    #      'tf': pd.Series(tpr - (1 - fpr), index=i), 'thresholds': pd.Series(thresholds, index=i)})
    # # roc.ix[(roc.tf - 0).abs().argsort()[:1]]
    # print(roc)
    # index = (roc.tf - 0).abs().argsort()[:1]
    # print(index)
    # print(roc.iloc[index])
    # print(thresholds)














