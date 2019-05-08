import pandas as pd
import numpy as np

def LoadData(filename):
    numFeat = len(open(filename).readline().split(','))-1
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines()[1:]:
        lineArr = []
        curLine = line.strip().split(',')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append([1.0]+lineArr)
        labelMat.append(int(curLine[-1]))
    return dataMat, labelMat

def sigmoid(inX):
    return 1.0/(1.0+np.exp(-inX))

def LogisticRegression(dataMatrix, classLabels):
    m,n = np.shape(dataMatrix)
    weights = np.ones((n,1))
    #  epoch = 100000
    epoch = 500
    for j in range(epoch): #迭代
        alpha = 0.01 + (0.01/(1+j))
        h = sigmoid(dataMatrix*weights)
        error = classLabels-h
        weights = weights+(alpha*dataMatrix.transpose()*error)/m
    #  print(weights)
    return weights

def score(ret, answer, num):
    cnt = 0
    tp = np.zeros((27,1))
    fp = np.zeros((27,1))
    fn = np.zeros((27,1))
    tn = np.zeros((27,1))
    for i in range(num):
        if ret[i] == answer[i]:
            cnt += 1
            tp[ret[i]] += 1
            for j in range(1,27) :
                if j != ret[i] :
                    tn[j] += 1
        else :
            fp[ret[i]] += 1
            for j in range(1,27) :
                if j == answer[i] :
                    fn[j] += 1
                else :
                    tn[j] += 1

    tp_sum = 0
    fp_sum = 0
    fn_sum = 0
    tn_sum = 0
    micro_precision = 0
    micro_recall = 0
    for i in range(1, 27):
        tp_sum += tp[i]
        fp_sum += fp[i]
        fn_sum += fn[i]
        tn_sum += tn[i]
    micro_precision = (tp_sum/(tp_sum+fp_sum))
    micro_recall = (tp_sum/(tp_sum+fn_sum))
    accuracy = (cnt/num)

    macro_precision = 0
    macro_recall = 0
    macro_F1 = 0
    for i in range(1, 27):
        macro_precision += (tp[i]/(tp[i]+fp[i]))
        macro_recall += (tp[i]/(tp[i]+fn[i]))
        macro_F1 += 2*((tp[i]/(tp[i]+fp[i]))*(tp[i]/(tp[i]+fn[i])))/((tp[i]/(tp[i]+fp[i]))+(tp[i]/(tp[i]+fn[i])))

    print("accuracy = %f"%accuracy)
    print("micro Precision = %f"%micro_precision )
    print("micro Recall = %f"%micro_recall )
    print("micro F1 = %f"%(2*(micro_precision*micro_recall)/(micro_precision+micro_recall)))
    print("macro Precision = %f"%(macro_precision/26))
    print("macro Recall = %f"%(macro_recall/26))
    print("macro F1 =% f"%(macro_F1/26))
    return

def vote(data_test, parameters):
    m, n = np.shape(data_test)
    ret = [0]*m
    for j in range(m):
        index = 0
        num = 0
        for i in range(26):
            tmp = sigmoid(data_test[j]*parameters[i])
            if tmp >=0 and tmp <= 1 and tmp > num:
                num = tmp
                index = i+1
            elif tmp<0 or tmp>1:
                print("Wrong probability")
        #  print("num = %f"%num)
        #  print("index = %d"%index)
        ret[j] = index
    return ret

def main():
    dataMat, labelMat = LoadData("train_set.csv")
    dataMat = np.mat(dataMat)
    labelMat = np.mat(labelMat).transpose()
    data_test, answer = LoadData("test_set.csv")
    data_test = np.mat(data_test)
    answer = np.mat(answer).transpose()

    parameters = []
    for i in range(1,27):
        labelMat_mo = np.where(labelMat == i, 1, 0)
        weights=LogisticRegression(dataMat, labelMat_mo)
        parameters.append(weights)
    m, n = np.shape(data_test)
    ret = vote(data_test, parameters)
    score(ret, answer, m)
    return

if __name__=='__main__':
    main()
