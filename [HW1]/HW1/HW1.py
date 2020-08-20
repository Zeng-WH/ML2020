import numpy as np
import pandas as pd
import sys
import math
import matplotlib.pyplot as plt
#Load 'train.csv'

            #value*/
#print(x_train)
#print(y_train)
def input_train_data(input_train_file):
    data = pd.read_csv(input_train_file, encoding='big5')
    data = data.iloc[:, 3:]
    data[data == 'NR'] = 0
    raw_data = data.to_numpy()
    #np.random.shuffle(raw_data)
    # 得到的数据为4320*18的资料，为依照每个月份重组成12个18*480的资料
    month_data = {}
    for month in range(12):
        sample = np.empty([18, 480])
        for day in range(20):
            sample[:, day * 24: (day + 1) * 24] = raw_data[18 * (20 * month + day): 18 * (20 * month + day + 1), :]
        month_data[month] = sample
    # 每个月会有480hrs, 每9小时形成一个data, 每个月会有471个data, 故总资料数
    # 为471*12笔，而每笔data有9*18的features
    # 对应的target即有471*12个
    x_train = np.empty([12 * 471, 18 * 9], dtype=float)
    y_train = np.empty([12 * 471, 1], dtype=float)
    for month in range(12):
        for day in range(20):
            for hour in range(24):
                if day == 19 and hour > 14:
                    continue
                x_train[month * 471 + day * 24 + hour, :] = month_data[month][:,
                                                            day * 24 + hour: day * 24 + hour + 9].reshape(1, -1)
                y_train[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9]
                # value
    np.save('x_train.npy', x_train)
    np.save('y_train.npy', y_train)
    #return x_train, y_train
def input_train_data_shuffle(input_train_file):
    data = pd.read_csv(input_train_file, encoding='big5')
    data = data.iloc[:, 3:]
    data[data == 'NR'] = 0
    raw_data = data.to_numpy()
    #np.random.shuffle(raw_data)
    # 得到的数据为4320*18的资料，为依照每个月份重组成12个18*480的资料
    month_data = {}
    for month in range(12):
        sample = np.empty([18, 480])
        for day in range(20):
            sample[:, day * 24: (day + 1) * 24] = raw_data[18 * (20 * month + day): 18 * (20 * month + day + 1), :]
        month_data[month] = sample
    # 每个月会有480hrs, 每9小时形成一个data, 每个月会有471个data, 故总资料数
    # 为471*12笔，而每笔data有9*18的features
    # 对应的target即有471*12个
    x_train = np.empty([12 * 471, 18 * 9], dtype=float)
    y_train = np.empty([12 * 471, 1], dtype=float)
    for month in range(12):
        for day in range(20):
            for hour in range(24):
                if day == 19 and hour > 14:
                    continue
                x_train[month * 471 + day * 24 + hour, :] = month_data[month][:,
                                                            day * 24 + hour: day * 24 + hour + 9].reshape(1, -1)
                y_train[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9]
                # value
    new_data = np.empty([12 * 471, 18 * 9+1], dtype=float)
    new_data[:,0:18*9]=x_train
    new_data[:,18*9]= y_train[:,0]
    np.random.shuffle(new_data)
    x_train=new_data[:,0:18*9]
    y_train[:,0]=new_data[:,18*9]
    np.save('x_train.npy', x_train)
    np.save('y_train.npy', y_train)
    #return x_train, y_train
#正则化处理
def Dataprocessor(x_train, y_train):
    mean_x_train = np.mean(x_train, axis=0)  # 18 * 9
    std_x_train = np.std(x_train, axis=0)
    for i in  range(len(x_train)): #12*471
        for j in range(len(x_train[0])): #18 * 9
            if std_x_train[j] != 0:
                x_train[i][j] = (x_train[i][j] - mean_x_train[j]) / std_x_train[j]
    xtrain = x_train[: math.floor(len(x_train) * 0.8), :]
    ytrain = y_train[: math.floor(len(y_train) * 0.8), :]
    xval = x_train[math.floor(len(x_train) * 0.8): , :]
    yval = y_train[math.floor(len(y_train) * 0.8): , :]
    np.save('xtrain.npy', xtrain)
    np.save('ytrain.npy', ytrain)
    np.save('xval.npy', xval)
    np.save('yval.npy', yval)
#learning_rate=100
#iter_time = 1000
#adagrad = np.zeros([np.size(x_train,1),1])
#eps=0.000000001
def train_model(xtrain, ytrain, learning_rate, iter_time, eps):
    m = np.size(xtrain, 0)
    #sample的个数
    n = np.size(xtrain, 1)
    #feature的个数
    w = np.zeros((n, 1))
    adagrad = 0
    b_adagrad = 0
    b = 0
    for t in range(iter_time):
        loss = np.sqrt(np.sum((np.dot(xtrain, w)+b*np.ones((m,1))- ytrain)**2)/m)
        if(t%100 ==0):
            print(str(t)+":"+str(loss))
        gradient = 2 * np.dot(xtrain.transpose(), np.dot(xtrain, w)+b*np.ones((m,1))- ytrain)
        b_gradient = 2 * np.sum(np.dot(xtrain, w)+b*np.ones((m,1))- ytrain)
        adagrad += gradient ** 2
        b_adagrad += b_gradient ** 2
        w = w - learning_rate * gradient/np.sqrt(adagrad+eps)
        b = b - learning_rate * b_gradient/np.sqrt(b_adagrad+eps)
    np.save('weight.npy', w)
    np.save('bias.npy', b)
def train_model1(xtrain, ytrain, learning_rate, iter_time):
    m = np.size(xtrain, 0)
    #sample的个数
    n = np.size(xtrain, 1)
    #feature的个数
    w = np.zeros((n, 1))
    b= 0
    for t in range(iter_time):
        loss = np.sqrt(np.sum((np.dot(xtrain, w)+b*np.ones((m,1))- ytrain)**2)/m)
        if(t%100 ==0):
            print(str(t)+":"+str(loss))
        gradient = 2 * np.dot(xtrain.transpose(), np.dot(xtrain, w)+b*np.ones((m,1))- ytrain)
        b_gradient = 2 * np.sum(np.dot(xtrain, w)+b*np.ones((m,1))- ytrain)
        w = w - learning_rate * gradient
        b = b - learning_rate * b_gradient
    np.save('weight.npy', w)
    np.save('bias.npy', b)
def train_model_adam(xtrain, ytrain, learning_rate, iter_time, eps, beta1, beta2):
    m = np.size(xtrain, 0)
    #sample的个数
    n = np.size(xtrain, 1)
    #feature的个数
    w = np.zeros((n,1))
    b = 0
    momentum = 0
    vs = 0
    b_vs = 0
    b_momentum = 0
    for t in range(1,iter_time):
        loss = np.sqrt(np.sum((np.dot(xtrain, w) + b * np.ones((m, 1)) - ytrain) ** 2) / m)
        if(t%100 ==0 ):
            print(str(t) + ":" + str(loss))
        gradient = 2 * np.dot(xtrain.transpose(), np.dot(xtrain, w) + b * np.ones((m, 1)) - ytrain)
        b_gradient = 2 * np.sum(np.dot(xtrain, w) + b * np.ones((m, 1)) - ytrain)
        vs = beta2 * vs + (1-beta2)*((gradient)**2)
        b_vs = beta2 * b_vs + (1-beta2)*((b_gradient)**2)
        momentum = beta1 * momentum + (1-beta1)*gradient
        b_momentum = beta1 * b_momentum + (1-beta1)*b_gradient
        w = w - ((learning_rate)/(np.sqrt(vs/(1-(beta2)**t))+eps)) * (momentum/(1-(beta1)**t))
        b = b - ((learning_rate)/(np.sqrt(b_vs/(1-(beta2)**t))+eps)) * (b_momentum/(1-(beta1)**t))
    np.save('weight.npy', w)
    np.save('bias.npy', b)
def train_model_adam_init(xtrain, ytrain, learning_rate, iter_time, eps, beta1, beta2):
    m = np.size(xtrain, 0)
    #sample的个数
    n = np.size(xtrain, 1)
    #feature的个数
    w = 1*np.random.randn(n,1)
    b = 1*np.random.randn()
    #w = np.zeros((n,1))
    #b = 0
    momentum = 0
    vs = 0
    b_vs = 0
    b_momentum = 0
    for t in range(1,iter_time):
        loss = np.sqrt(np.sum((np.dot(xtrain, w) + b * np.ones((m, 1)) - ytrain) ** 2) / m)
        if(t%100 ==0 ):
            print(str(t) + ":" + str(loss))
        gradient = 2 * np.dot(xtrain.transpose(), np.dot(xtrain, w) + b * np.ones((m, 1)) - ytrain)
        b_gradient = 2 * np.sum(np.dot(xtrain, w) + b * np.ones((m, 1)) - ytrain)
        vs = beta2 * vs + (1-beta2)*((gradient)**2)
        b_vs = beta2 * b_vs + (1-beta2)*((b_gradient)**2)
        momentum = beta1 * momentum + (1-beta1)*gradient
        b_momentum = beta1 * b_momentum + (1-beta1)*b_gradient
        w = w - ((learning_rate)/(np.sqrt(vs/(1-(beta2)**t))+eps)) * (momentum/(1-(beta1)**t))
        b = b - ((learning_rate)/(np.sqrt(b_vs/(1-(beta2)**t))+eps)) * (b_momentum/(1-(beta1)**t))
    np.save('weight.npy', w)
    np.save('bias.npy', b)
def eval_model(xval, yval, w, b):
    m = np.size(xval, 0)
    #eval的sample的数目
    n = np.size(xval, 1)
    y_hat=np.dot(xval, w)+b*np.ones((m,1))
    #feature的数目
    eval_loss = np.sqrt(np.sum(np.power(np.dot(xval, w)+b*np.ones((m,1))- yval, 2))/m)
    np.save('yhat.npy',y_hat)
    return eval_loss
def main( ):
    input_train_data_shuffle('./train.csv')
    print('The data has been loaded')
    x_train = np.load('x_train.npy')
    y_train = np.load('y_train.npy')
    Dataprocessor(x_train, y_train)
    print('The data has been processed')
    xtrain = np.load('xtrain.npy')
    ytrain = np.load('ytrain.npy')
    xval = np.load('xval.npy')
    yval = np.load('yval.npy')
    learning_rate = 1
    iter_time =1000
    eps = 0.000000001
    #train_model(xtrain, ytrain, learning_rate, iter_time, eps)
    train_model_adam_init(xtrain, ytrain, learning_rate, iter_time, eps, 0.99, 0.999)
    print('The model has been trained')
    w = np.load('weight.npy')
    print('the value of w:'+str(w))
    b = np.load('bias.npy')
    print('eval the model')
    eval_loss=eval_model(xval, yval, w, b)
    #print('eval the model')
    print('eval loss:'+str(eval_loss))
    #print(np.random.randn(10,1))
if __name__ == "__main__":
    main()




