import numpy as np
from collections import Counter


def KNN(matTrain, y_train, matTest, k=50):
    numTrain = matTrain.shape[0]
    distances = [np.linalg.norm(matTrain[i] - matTest) for i in range(numTrain)]
    # print(distances)
    distances = np.array(distances)
    nearest = distances.argsort()
    topK_y = [y_train[i] for i in nearest[:k]]
    votes = Counter(topK_y)
    predict_y = votes.most_common(1)[0][0]
    return predict_y
