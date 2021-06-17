import numpy as np
from collections import Counter


def KNN(matTrain, y_train, matTest, k=50):
    numTrain = matTrain.shape[0]
    numTest = matTest.shape[0]
    for i in range(numTest):
        distances = [np.linalg.norm(matTrain[i] - matTest) for i in range(numTrain)]
        # print(distances)
        distances = np.array(distances)
        nearest = distances.argsort()  # 返回原数组的下标
        topK_y = [y_train[i] for i in nearest[:k]]  # topK_y得到最近的k个点所属的类别（很多个）
        votes = Counter(topK_y)
        predict_y = votes.most_common(1)[0][0]  # 多个类别中的大多数类别
    return predict_y
