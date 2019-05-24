import pandas as pd
import numpy as np
import sklearn
from sklearn import tree

class RandomForest():
    """Random Forest classifier. Uses a collection of classification trees that
    trains on random subsets of the data using a random subsets of the features.
    Parameters:
    -----------
    n_estimators: int
        The number of classification trees that are used.
    max_features: int
        The maximum number of features that the classification trees are allowed to
        use.
    min_samples_split: int
        The minimum number of samples needed to make a split when building a tree.
    min_gain: float
        The minimum impurity required to split the tree further.
    max_depth: int
        The maximum depth of a tree.
    """
    def __init__(self, n_estimators=2000, min_samples_split=2, min_gain=0, max_features=None):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.max_features = max_features

        self.trees = []
        # bulid forest
        for _ in range(self.n_estimators):
            tree_s = tree.DecisionTreeRegressor(min_samples_split=self.min_samples_split, min_impurity_decrease=self.min_gain, splitter='random')
            self.trees.append(tree_s)

    def fit(self, X, Y):
        sub_sets = self.get_bootstrap_data(X, Y)
        n_features = X.shape[1]
        if self.max_features == None:
            self.max_features = int(np.sqrt(n_features))
        for i in range(self.n_estimators):
            sub_X, sub_Y = sub_sets[i]
            idx = np.random.choice(n_features, self.max_features, replace=True)
            sub_X = sub_X[:, idx]
            self.trees[i].fit(sub_X, sub_Y)
            self.trees[i].feature_indices = idx
            print("tree", i, "fit complete")

    def predict(self, X):
        y_preds = []
        for i in range(self.n_estimators):
            idx = self.trees[i].feature_indices
            sub_X = X[:, idx]
            y_pre = self.trees[i].predict(sub_X)
            y_preds.append(y_pre)
        y_preds = np.array(y_preds).T
        y_pred = []
        for y_p in y_preds:
            y_pred.append(y_p.mean())
        return y_pred

    def get_bootstrap_data(self, X, Y):
        m = X.shape[0]
        Y = Y.reshape(m, 1)

        X_Y = np.hstack((X, Y))
        np.random.shuffle(X_Y)

        data_sets = []
        for _ in range(self.n_estimators):
            idm = np.random.choice(m, m, replace=True)
            bootstrap_X_Y = X_Y[idm, :]
            bootstrap_X = bootstrap_X_Y[:, :-1]
            bootstrap_Y = bootstrap_X_Y[:, -1:]
            data_sets.append([bootstrap_X, bootstrap_Y])
        return data_sets

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
    modi = np.vstack((prob, labels))
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

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    clf = RandomForest()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    length = len(pred)
    y_test = y_test.T
    print(pred)
    print(y_test)

    print("AUC = ",calAUC(pred, y_test, length))
