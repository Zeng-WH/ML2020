import numpy as np
import pandas as pd
import math
import csv
testdata = pd.read_csv('./test.csv', header = None, encoding = 'big5')
testdata = testdata.iloc[:, 2:]
testdata[testdata == 'NR'] = 0
testdata = testdata.to_numpy()
#print(np.size(test_data))
test_x = np.empty([240, 18*9], dtype = float)
for i in range(240):
    test_x[i, :] = testdata[18 * i: 18* (i + 1), :].reshape(1, -1)
#print(np.size(test_x,0))
#print(len(test_x))
mean_x = np.mean(test_x, axis=0)  # 18 * 9
#print(np.size(mean_x,0))
std_x = np.std(test_x, axis=0)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
w = np.load('weight.npy')
b = np.load('bias.npy')
ans_y = np.dot(test_x, w) + b*np.ones((240,1))
with open('submit1.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        print(row)