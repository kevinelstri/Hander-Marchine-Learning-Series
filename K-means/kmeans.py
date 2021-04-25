# by kevinelstri 2021/04/22
# 手写kmeans算法，比《机器学习实战》算法更好理解
import math
import random


# 读取数据
def load_data(filename):
    dataSet = []
    for line in open(filename).readlines():
        dataline = [float(i) for i in line.strip().split('\t')]
        dataSet.append(dataline)
    return dataSet


# 计算距离(欧式距离)
def dist(vec1, vec2):
    return math.sqrt(pow(vec1[0] - vec2[0], 2) + pow(vec1[1] - vec2[1], 2))


# 计算平均值例子
def mean():
    L = [[1.658985, 4.285136], [-3.453687, 3.424321], [4.838138, -1.151539]]
    sum_list = [sum(i) for i in zip(*L)]
    avg_list = [sum(i) / len(L) for i in zip(*L)]
    return sum_list, avg_list


# 随机获取k个聚类中心
def randomCenter(dataSet, k):
    dataLength = len(dataSet)
    if k > dataLength:
        return
    randomCenter = []
    for i in range(k):
        randomNum = random.randint(1, dataLength - 1)  # 获取随机整数
        randomCenter.append(dataSet[randomNum])
    return randomCenter


# 聚类算法
def kmeans(dataSet, randomCenter, k):
    randomCenterStart = randomCenter  # 初识聚类中心
    randomCenterEnd = []
    cluster = [[] for i in range(k)]
    dataLength = len(dataSet)
    for i in range(dataLength):
        distList = []
        for j in range(k):
            distList.append(dist(dataSet[i], randomCenter[j]))
        cluster[distList.index(min(distList))].append(dataSet[i])  # 计算最小距离，进行分类，每个类别一个list

    for i in range(k):
        tmp = cluster[i]
        avg = [sum(i) / len(tmp) for i in zip(*tmp)]  # 二维list计算平均值
        randomCenterEnd.append(avg)  # 聚类之后的聚类中心
    # print('old center', randomCenterStart)
    # print('new center:', randomCenterEnd)

    if randomCenterStart != randomCenterEnd:  # 聚类中心不在改变，则聚类完成
        kmeans(dataSet, randomCenterEnd, k)  # 递归实现聚类

    return cluster, randomCenterEnd


# 主函数调用
if __name__ == '__main__':
    # 读取数据集
    filename = 'testSet.txt'
    dataSet = load_data(filename)
    print(f'数据集:{dataSet}')

    # 随机获取k个聚类中心
    k = 3
    randomCenter = randomCenter(dataSet, k)
    print(f'随机{k}个聚类中心:{randomCenter}')

    # 聚类算法
    cluster, randomCenter = kmeans(dataSet, randomCenter, k)

    # 打印聚类结果
    for i in range(k):
        print(f'第{i + 1}个聚类中心：', randomCenter[i])
        print(f'第{i + 1}个聚类结果：', cluster[i])
