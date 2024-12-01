import csv
import numpy as np
from sklearn.preprocessing import StandardScaler

def readData(filename, hd=True):
    data, header = [], None
    with open(filename, 'r', encoding='utf-8') as csvfile:
        it = csv.reader(csvfile, delimiter=',')
        if hd == True:
            header = it.__next__()
        for row in it:
            data.append(row)

    return np.array(data), np.array(header)

def getData(data_path):
    data, _ = readData(data_path, False)
    data = (data[1:]).astype(float)
    X, Y = data[:, 0:-1], data[:, -1].astype(int)

    # 对训练数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    for i in range(len(Y)):
        if(Y[i] == 0):
            Y[i] = -1
    return X, Y