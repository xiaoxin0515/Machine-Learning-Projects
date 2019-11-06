import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import pdist, squareform


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def all_data_by_label(label):
    labels_ = []
    data_ = []
    for i in range(1, 6):
        labels_ += unpickle("cifar-10-batches-py/data_batch_%d"%i)[b"labels"]
        data_.append(unpickle("cifar-10-batches-py/data_batch_%d" % i)[b"data"])
    labels_ += unpickle("cifar-10-batches-py/test_batch")[b"labels"]
    data_.append(unpickle("cifar-10-batches-py/test_batch")[b"data"])
    data = np.concatenate(data_, 0)
    labels = np.array(labels_)

    data_select_ = []
    for i in range(len(labels)):
        if(labels[i] == label):
            data_select_.append(data[i])
    data_select = np.array(data_select_)

    return data_select

def get_rgb(pixel):
    assert len(pixel) == 3072
    pixel = pixel.astype(int)
    r = pixel[0:1024]
    r = np.reshape(r, [32, 32, 1])
    g = pixel[1024:2048]
    g = np.reshape(g, [32, 32, 1])
    b = pixel[2048:3072]
    b = np.reshape(b, [32, 32, 1])

    rgb = np.concatenate([r, g, b], -1)

    return rgb

def squared_difference_mean(data1, data2):
    distance = euclidean_distances(data1, data2)
    num = distance.shape[0]
    error = 0
    for i in range(num):
        error += distance[i][i]**2
    error_mean = error/num

    return error_mean

def MDS(matrix, k):
    I = np.identity(10)
    ones = 1/10 * np.ones([10, 10])
    A = I - ones
    W = -1/2 * A * matrix * A.T
    eigValue, eigVec = np.linalg.eig(W)
    sort_index = np.argsort(-eigValue)
    selectVec = np.matrix(eigVec.T[:k])
    U = selectVec.T
    lam_diag = eigValue[sort_index]
    lam_select_diag = np.sqrt(lam_diag[:k])
    lam = np.diag(lam_select_diag)
    matrix_scale = U * lam

    return matrix_scale

def get_eigenVector(data, mean, component_num):
    row = data.shape[0]
    trans_mean = data - np.tile(mean, (row, 1))
    covmat = np.cov(trans_mean.T)
    eigValue, eigVec = np.linalg.eig(covmat)
    sort_index = np.argsort(-eigValue)
    selectVec = np.matrix(eigVec.T[:component_num])
    U = selectVec.T

    return U

def inverse(data, mean, u):
    row = data.shape[0]
    trans_mean = data - np.tile(mean, (row, 1))
    new = np.dot(u.T, trans_mean.T)  ## np.dot: the latter one multiplies the former one
    new = np.dot(u, new)
    new = new.T + np.tile(mean, (row, 1))

    return new


if __name__ == '__main__':
    # file_name = unpickle("cifar-10-batches-py/batches.meta")
    # label_name = file_name[b"label_names"]
    label_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # error = []
    # data_list = []
    # mean_matrix = np.zeros(shape=(10, 3072))
    # for i in range(10):
    #     data = all_data_by_label(i)
    #     # compute mean picture
    #     data_list.append(data)
    #     mean = np.mean(data, axis=0)
    #     mean_matrix[i] = mean
    #     rgb_mean = get_rgb(mean)
    #     plt.subplot(2, 5, (1+i))
    #     plt.imshow(rgb_mean)
    #     plt.axis("off")
    # #plt.show()
    #
    #     # 20 components pca
    #     row = data.shape[0]
    #     trans_mean = data - np.tile(mean, (row, 1))
    #     pca_function = PCA(n_components=20)
    #     data_new = pca_function.fit_transform(trans_mean)
    #     data_inverse = pca_function.inverse_transform(data_new) + np.tile(mean, (row, 1))
    #     # compute error
    #     error_mean = squared_difference_mean(data, data_inverse)
    #     error.append(error_mean)
    # plt.savefig("m.png")
    # # try
    # # print(type(data_list[0]))
    # # print(data_list[0].shape)
    # # u = get_eigenVector(data_list[0], mean_matrix[0], 20)
    # # new = inverse(data_list[0], mean_matrix[0], u)
    #
    # # mean square distance matrix
    # distance_matrix = euclidean_distances(mean_matrix, mean_matrix) ** 2
    # np.savetxt('partb_distances.csv', distance_matrix, delimiter=',')
    #
    #
    # fig1 = plt.figure("fig1")
    # plt.title("difference between the pca reconstructed and original image")
    # plt.bar(label_name, error)
    # plt.xlabel("label name")
    # plt.ylabel("difference")
    # print(error)
    # fig1.savefig("difference")

    # principal coordinate analysis using euclidean distance
    distance_matrix = np.matrix(np.loadtxt("correct/partb_distances.csv", delimiter=","))
    scale_matrix = MDS(distance_matrix, 2)
    component1 = np.array(scale_matrix.T[0])
    component2 = np.array(scale_matrix.T[1])
    component1 = component1[0]
    component2 = component2[0]
    fig2 = plt.figure("fig2")
    plt.title("principal coordinate analysis using euclidean distance")
    plt.scatter(component1, component2)
    plt.xlabel("component1")
    plt.ylabel("component2")
    for i, label in enumerate(label_name):
        plt.annotate(label, (component1[i], component2[i]))
    fig2.savefig("pcoa_euclidean")

    # ###### Part C #########
    # distance_partC = np.zeros(shape=(10, 10))
    # for i in range(10):
    #     temp_list = []
    #     data1 = data_list[i]
    #     mean1 = mean_matrix[i]
    #     u1 = get_eigenVector(data1, mean1,  20)
    #     for j in range(10):
    #         data2 = data_list[j]
    #         mean2 = mean_matrix[j]
    #         u2 = get_eigenVector(data2, mean2, 20)
    #         E_a_b = squared_difference_mean(data1, inverse(data1, mean1, u2))
    #         E_b_a = squared_difference_mean(data2, inverse(data2, mean2, u1))
    #         ########### there should be a sanity check
    #
    #         E_a_to_b = (E_a_b + E_b_a)/2
    #         temp_list.append(E_a_to_b)
    #     print(temp_list)
    #     distance_partC[i] = np.array(temp_list)
    # np.savetxt("partc_distances.csv", distance_partC, delimiter=",")
    #
    #
    # #  do MDS and then plot
    # # distance_partC = np.matrix(np.loadtxt("correct/partc_distances.csv", delimiter=","))
    # scale_matrix_partC = MDS(distance_partC, 2)
    # # print(scale_matrix)
    # # print(scale_matrix.shape)
    # component1 = np.array(scale_matrix_partC.T[0])
    # component2 = np.array(scale_matrix_partC.T[1])
    # component1 = component1[0]
    # component2 = component2[0]
    # # print(component1.shape)
    # # print(component2.shape)
    # fig3 = plt.figure("fig3")
    # plt.title("principal coordinate analysis using the similarity metric")
    # plt.scatter(component1, component2)
    # plt.xlabel("component1")
    # plt.ylabel("component2")
    # for i, label in enumerate(label_name):
    #     plt.annotate(label, (component1[i], component2[i]))
    # fig3.savefig("partC")
    #



