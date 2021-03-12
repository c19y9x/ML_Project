from numpy import *

def loadDataSet():#打开文本文件testSet.txt并逐行读取
    #每行前两个值分别是X1和X2，第三个值是数据对应的类别标签
    dataMat = []; labelMat = []
    fr = open('ch5\\data\\testSet.txt')#打开文本文件testSet.txt
    for line in fr.readlines():#逐行读取
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])#存放数据X1和X2
        labelMat.append(int(lineArr[2]))#存放第三个值标签
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

# 梯度上升优化算法
def gradAscent(dataMatIn,classLabels):#dataMatIn是一个2维Numpy数组，每列分别代表每个不同的特征，每行则代表每个训练样本
    dataMatrix = mat(dataMatIn)#mat创建矩阵
    labelMat = mat(classLabels).transpose()#为了方便矩阵进行计算，将原向量进行转置
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))#构建n行1列的全1矩阵
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)#计算真实类别与预测类别的差值，按照差值的方向调整回归系数
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]#dataArr有几行
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])#1为正样本
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i, 2])#0为负样本
    fig = plt.figure() #创建一个新图形
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')#绘制正样本
    ax.scatter(xcord2, ycord2, s=30, c='green')#绘制负样本
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]#最佳拟合直线
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

# 随机梯度算法
def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   #initialize to all ones
    for i in range(m):
        # h,error全是数值，没有矩阵转换过程。
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

# 改进的随机梯度算法
def stocGradAscent1(dataMatrix,classLabels,numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)
    """
    i 是样本点的下标,j 是迭代次数
    """
    for j in range(numIter):         
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0 + j + i) + 0.01  #alpha每次迭代时需要调整
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha *error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights


dataMat,labelMat = loadDataSet()
print(stocGradAscent0(array(dataMat),labelMat))
plotBestFit(stocGradAscent0(array(dataMat),labelMat)) #getA()将weights矩阵转换为数组，getA()函数与mat()函数的功能相反  
""" 
如果是矩阵的话会报这样的错： 
"have shapes {} and {}".format(x.shape, y.shape)) 
ValueError: x and y must have same first dimension, but have shapes (60,) and (1, 60) 
为啥要用数组呢？因为 x = arange(-3.0, 3.0, 0.1)，len(x) = [3-(-3)]/0.1 = 60 
而weights是矩阵的话，y = (-weights[0]-weights[1]*x)/weights[2]，len(y) = 1，有60个x，y只有一个，你这样都画不了线 
而weights是数据的话，len(y) = 60 
""" 
