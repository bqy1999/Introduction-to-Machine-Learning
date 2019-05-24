##########################################################
# Author        : ArthurBi
# Email         : Arthurbqy1999@gmail.com
# Last modified : 2019-05-21 20:01
# Filename      : AdaBoost.py
# Description   : Pls contact me if you have questions.
##########################################################
import pandas as pd
import numpy as np
import urllib
import sklearn
from sklearn import tree

def maxminnorm(array):
    maxcols=array.max(axis=0)
    mincols=array.min(axis=0)
    #  print(maxcols)
    #  print(mincols)
    array = (array-mincols)/(maxcols-mincols)
    #  t=(array-mincols[0])/(maxcols[0]-mincols[0])
    return array

def calAUC(prob, labels, length):
    '''
计算AUC并返回
:param prob: 模型预测样本为Positive的概率列表
:param labels: 样本的真实类别列表，其中1表示Positive，0表示Negtive
:return: AUC，类型为float
'''
    M = 0
    N = 0
    R = 0
    #  print(prob)
    #  print(labels)
    modi = np.vstack((prob, labels))
    #  print("modi")
    #  print(modi)
    modi = modi.T[np.lexsort(modi[::-1,:])].T
    #  print("modi")
    #  print(modi)
    #  print("length")
    #  print(len(labels))
    for i in range(0, length):
        if modi[1][i] == 0 :
            N = N+1
        else :
            M = M+1
            R = R+i+1
    return float((R*1.0-M*1.0*(M*1.0+1.0)/2.0)/(M*N*1.0))

def Adaboost_clf(Y_train, X_train, Y_test, X_test, M=2000, weak_clf=tree.DecisionTreeClassifier(max_depth = 1, random_state = 1)):
    n_train, n_test = len(X_train), len(X_test)
    #  print("len of train =",n_train)
    #  print("len of test =",n_test)
    # Initialize weights
    w = np.ones(n_train) / n_train
    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    for i in range(M):
        # Fit a classifier with the specific weights
        weak_clf.fit(X_train, Y_train, sample_weight = w)
        pred_train_i = np.array(weak_clf.predict(X_train))
        pred_test_i = np.array(weak_clf.predict(X_test))
        #  print("pred_train", i)
        #  print(pred_train_i)
        #  print("Y_train")
        #  print(Y_train[0])

        # Indicator function
        miss = [int(x) for x in (np.abs(pred_train_i - Y_train.T[0])>0.5)]
        #  print("miss =",miss)
        #  print(pred_train_i)
        #  print(Y_train)
        #  print("miss length=", len(miss))
        print("weak_clf_%02d train acc: %.4f"
                % (i + 1, 1 - sum(miss) / n_train))

        # Error
        err_m = np.dot(w, miss)
        # Alpha
        alpha_m = 0.5 * np.log((1 - err_m) / float(err_m))
        # New weights
        miss2 = [x if x==1 else -1 for x in miss] # -1 * y_i * G(x_i): 1 / -1
        w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2]))
        w = w / sum(w)

        # Add to prediction
        pred_train_i = [1 if x > 0.5 else -1 for x in pred_train_i]
        pred_test_i = [1 if x > 0.5 else -1 for x in pred_test_i]
        pred_train = pred_train + np.multiply(alpha_m, pred_train_i)
        pred_test = pred_test + np.multiply(alpha_m, pred_test_i)

    #  pred_train = (pred_train > 0) * 1
    #  pred_test = (pred_test > 0) * 1
    pred_train = maxminnorm(pred_train)
    pred_test = maxminnorm(pred_test)
    #  print(pred_train)
    #  print(pred_test)
    print("AUC = ",calAUC(pred_test, Y_test.T[0], n_test))
    return

if __name__ == '__main__':
    fd1 = pd.read_csv("adult.data", sep=',', header=None)
    fd2 = pd.read_csv("adult.test", sep=',', header=None)

    X_train = fd1.iloc[:,0:13]
    y_train = fd1.iloc[:,14]
    X_test = fd2.iloc[:,0:13]
    y_test = fd2.iloc[:,14]

    X = pd.concat([X_train, X_test], axis=0)
    X = pd.get_dummies(X)
    X = X.reset_index()
    X_train = X.iloc[0:32561]
    X_test = X.iloc[32561:]
    X_test = X_test.reset_index()
    X_train = X_train.drop('index', axis=1)
    X_test = X_test.drop('level_0', axis=1)
    X_test = X_test.drop('index', axis=1)

    y_train = pd.get_dummies(y_train)
    y_train.columns = ['less', 'more']
    y_train = y_train.drop('less', axis=1)
    y_test = pd.get_dummies(y_test)
    y_test.columns = ['less', 'more']
    y_test = y_test.drop('less', axis=1)

    #  y_train = y_train.values
    #  y_test = y_test.values
    #  print(y_train.T[0])
    #  print(y_test.T[0])
    Adaboost_clf(y_train, X_train, y_test, X_test)
