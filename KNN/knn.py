import numpy as np
import math
from collections import Counter


# 读取数据
def load_data(filename):
    initMat = []
    initLab = []
    for line in open(filename).readlines():
        line = [float(i) for i in line.strip().split('\t')]
        initMat.append(line[:3])  # 分类数据 list格式
        initLab.append(line[-1])  # 分类标签
    npInitMat = np.array(initMat)  # 分类数据 np.array格式，更加归整
    return initMat, initLab, npInitMat


# 归一化计算
# norm = (value-min)/(max-min)
def Norm(dataSet):
    minValues = dataSet.min(0)  # np矩阵计算最小值
    maxValues = dataSet.max(0)  # np矩阵计算最大值
    return (dataSet - minValues) / (maxValues - minValues)


# 划分训练集和测试集(8:2)
def splitTrainTest(dataSet):
    totalLen = len(dataSet)
    trainSet = dataSet[:int(totalLen * 0.8)]
    testSet = dataSet[:int(totalLen * 0.2)]
    # trainLabel = trainSet[:, -1:]
    # testLabel = testSet[:, -1:]
    return trainSet, testSet  # , trainLabel, testLabel


# 欧式距离
def dist(vec1, vec2):
    return math.sqrt(sum(pow(vec1 - vec2, 2)))


# knn算法
def knn_classify(trainSet, testSet, trainLabel, k):
    testLabels = []
    for testVec in testSet:
        distList = []
        for trainVec in trainSet:
            distList.append(dist(testVec, trainVec))  # 计算单个测试向量与单个训练向量之间的距离
        distList_k = sorted(distList)[:k]  # 距离从小到大排序，获取前k个
        distList_k_lab = [trainLabel[distList.index(data_k)] for data_k in distList_k]  # 获取最小的k个元素的标签
        result = Counter(distList_k_lab)  # 计算元素个数
        maxValue = sorted(result.items(), reverse=True)[0][0]  # 获取出现次数最多的标签
        testLabels.append(maxValue)  # 出现次数最多的标签作为k近邻标签
    return testLabels


# 计算错误率
def errorRate(testLabels, actualLabels):
    errorCount = 0
    totalCount = len(testLabels)
    for i in range(totalCount):
        if testLabels[i] != actualLabels[i]:
            errorCount += 1
    return float(errorCount / totalCount)


# 主函数调用
if __name__ == '__main__':
    # 读取数据
    filename = 'datingTestSet2.txt'
    initMat, initLab, npInitMat = load_data(filename)
    print('分类数据(list)：\n', initMat)  # 分类数据
    print('分类标签：\n', initLab)  # 分类标签
    print('分类数据(np.array)：\n', npInitMat)

    # 归一化计算
    normMat = Norm(npInitMat)
    print('分类数据(np.array)归一化结果：\n', normMat)

    # 划分训练集和测试集
    trainSet, testSet = splitTrainTest(normMat)
    print('训练集：\n', trainSet)
    print('训练集长度：\n', len(trainSet))
    print('测试集：\n', testSet)
    print('测试集长度：\n', len(testSet))

    trainLabel = initLab[:int(len(initLab) * 0.8)]
    actualLabel = initLab[:int(len(initLab) * 0.2)]
    print('训练集标签：\n', trainLabel)
    print('训练集标签个数：\n', len(trainLabel))
    print('测试集实际标签：\n', actualLabel)
    print('测试集实际标签个数：\n', len(actualLabel))

    # knn算法
    k = 3
    testLabel = knn_classify(trainSet, testSet, trainLabel, k)
    print('测试集结果标签: \n', testLabel)

    # 错误率
    error_Rate = errorRate(testLabel, actualLabel)
    print('错误率：\n', error_Rate)
