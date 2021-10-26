'''
编写感知机算法
'''


import numpy as np
import time


def loadData(fileName):
    '''
    加载Mnist数据集
    :param fileName:要加载的数据集路径
    :return: list形式的数据集及标记
    '''
    print('Start to read data')
    dataArr = []
    labelArr = []

    # Mnsit有0-9是个标记，由于是二分类任务，所以将>=5的作为1，<5为-1
    # data[0]为标签，data[1:]为数据
    '''
        编写数据集处理算法
    '''
    with open(fileName, 'r') as fr:
        for f in fr:
            # 处理label,大于5为1,小于等于5为0
            num = int(f[0])
            if num > 5:
                labelArr.append(1)
            else:
                labelArr.append(-1)
            # 处理其余数据,按列添加至dataArr中
            temp = []
            f = eval(f)
            for i in range(1, len(f)):
                temp.append(int(f[i]))
            dataArr.append(temp)
    fr.close()
    return dataArr, labelArr


def perceptron(dataArr, labelArr, iter=50):
    '''
    感知器训练过程
    :param dataArr:训练集的数据 (list)
    :param labelArr: 训练集的标签 (list)
    :param iter: 迭代次数，默认50
    :return: 训练好的w和b
    '''
    print('Start to train')
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).T
    # 获取数据矩阵的大小为m*n
    m, n = np.shape(dataMat)
    # 创建初始权重w，初始值全为0
    # np.shape(dataMat)的返回值为m，n -> np.shape(dataMat)[1])的值即为n，与样本长度保持一致
    w = np.zeros((1, np.shape(dataMat)[1]))
    # 初始化偏置b为0
    b = 0
    # 初始化步长，也就是梯度下降过程中的n，控制梯度下降速率,默认h=0.0001
    h = 1

    # 进行iter次迭代计算，以及梯度下降
    '''
        编写感知机核心算法
    '''
    for i in range(iter):
        for j in range(m):
            data = np.array(dataMat[j])
            # 错误：若使用y = labelMat[j][0]，y仍然是<class 'numpy.matrix'>
            y = labelMat[j, 0]
            # 迭代更新w和b
            if y * (w.dot(data.T) + b) <= 0:
                w = w + data * y * h
                b = b + h * y
                break
    return w, b


def model_test(dataArr, labelArr, w, b):
    '''
    测试准确率
    :param dataArr:测试集
    :param labelArr: 测试集标签
    :param w: 训练获得的权重w
    :param b: 训练获得的偏置b
    :return: 正确率
    '''
    print('Start to test')
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).T
    # 获取数据矩阵的大小为m*n
    m, n = np.shape(dataMat)
    # 错误样本数计数
    errorCnt = 0
    # 遍历所有测试样本, 统计样本分类错误数
    '''
        编写测试算法
    '''
    for i in range(m):
        data = np.array(dataMat[i]).T
        y = labelMat[i, 0]
        if y * (w.dot(data) + b) <= 0:
            errorCnt += 1
    # 正确率 = 1 - （样本分类错误数 / 样本总数）
    print('errorCnt=', errorCnt, 'm=', m)
    accuracy = 1 - (errorCnt / m)
    return accuracy


if __name__ == '__main__':
    # 获取当前时间，作为开始时间
    start = time.time()

    # 获取训练集及标签
    trainData, trainLabel = loadData('./Mnist/mnist_train.csv')
    # 获取测试集及标签
    testData, testLabel = loadData('./Mnist/mnist_test.csv')

    # 训练获得权重及偏置
    w, b = perceptron(trainData, trainLabel, iter=50)
    # 进行测试，获得正确率
    accuracy = model_test(testData, testLabel, w, b)

    # 获取当前时间，作为结束时间
    end = time.time()
    # 打印正确率
    print('Accuracy is:', accuracy)
    # 打印用时时长
    print('Total time:', end - start)
