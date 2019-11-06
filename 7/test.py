import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import pickle
from sklearn.feature_extraction import stop_words
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial

# dataSetI = [3, 45, 7, 2]
# dataSetII = [2, 54, 13, 15]
# a = [dataSetI, dataSetII]
# result = cosine_similarity(a)
# print(result)
#
# result1 = 1 - spatial.distance.cosine(dataSetI, dataSetII)
# print(result1)

# print(stop_words.ENGLISH_STOP_WORDS)

data_input1 = open('stop_list.pkl', 'rb')
a = pickle.load(data_input1)
data_input1.close()

print(a)
# with open('stop_list.txt', 'w') as file:
#     for i in a:
#         print(i)
# file.close()

# print(len(a))
#
# plt.scatter(range(len(a)), a)
# plt.savefig('a')
# plt.show()




# text = ['ae,be,ce', 'be,ce,de', 'ee,ke,ae', 'le,pe,me']
# text = ['Horrible customer service',]
# cv = CountVectorizer(stop_words=None, lowercase=True)
# # # print(cv.fit(text))
# result = cv.fit_transform(text)
# count = result.toarray()
# # s = count.sum(axis=0)
# print(count)
# print(s)
# count_dict = cv.vocabulary_
# print(count_dict)
# value = count_dict.values()
# print(value)
#
# count_ordered = sorted(value, reverse=True)
# print(count_ordered)
#
# plt.scatter(range(len(count_ordered)), count_ordered)
# plt.show()

# a = 'python is an important language'
# print(a[0])