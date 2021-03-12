import numpy as np
import operator

from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

def createDataSet ():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]]) 
    labels = ['A','A','B','B'] 
    return group,labels

#inX待预测向量
def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0] #取出训练集矩阵的行数，即样本个数
    diffMat = np.tile(inX,(dataSetSize,1))-dataSet #1.将inX扩展成dataSetSize行1列的矩阵2.将扩展后的矩阵与dataSet相减，得到坐标之间的差值
    sqDiffMat = diffMat**2 #求diffMat中每个元素的平方
    sqDistances = sqDiffMat.sum(axis=1) #因为axis为1，所以求的是每个行中所有元素的和（axis=1：行运算；axis=0：列运算）
    distances = sqDistances**0.5#开根号，求inX与样本中每个元素的距离
    sortedDistIndicies = distances.argsort()#将距离由小到大排序，返回在已经排好序的矩阵中各元素在原来矩阵中的原始位置
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]] #获取与inX相对最小距离的元素在其原序列中的位置，并取出其对应的标签
        #get是取字典里的元素，如果之前这个voteIlabel是有的，那么就返回字典里这个voteIlabel键里的值，如果没有就返回0
        #算出离目标点距离最近的k个点的类别，这个点是哪个类别哪个类别就加1
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1 #可以自动增加元素
    #key=operator.itemgetter(1)的意思是按照字典里的‘值’进行排序，classCount字典里的数据样式类似于“{A:1,B:3}”。reverse=True是降序排序。
    #key=operator.itemgetter(0)表示按照字典里的键进行排序。
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True) 
    return sortedClassCount[0][0] #返回类别最多的类别


# 函数说明:打开并解析文件，对数据进行分类：1代表不喜欢,2代表魅力一般,3代表极具魅力
 
# Parameters:
#     filename - 文件名
# Returns:
#     returnMat - 特征矩阵
#     classLabelVector - 分类Label向量

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #按行读取
    returnMat = np.zeros((numberOfLines,3))        #生成一个行数*3的零矩阵
    classLabelVector = []                       #prepare labels return   
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1
    return returnMat,classLabelVector

#     数据可视化
#     Parameters:
#     datingDataMat - 特征矩阵
#     datingLabels - 分类Label
# Returns:
#     无

def showdatas(datingDataMat, datingLabels):
    #设置汉字格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    #将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    #当nrow=2,nclos=2时,代表fig画布被分为四个区域,axs[0][0]表示第一行第一个区域
    fig, axs = plt.subplots(nrows=2, ncols=2,sharex=False, sharey=False, figsize=(15,8))
 
    numberOfLabels = len(datingLabels)
    LabelsColors = []
    for i in datingLabels:
        if i == 1:
            LabelsColors.append('black')
        if i == 2:
            LabelsColors.append('orange')
        if i == 3:
            LabelsColors.append('red')
    #画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第二列(玩游戏)数据画散点数据,散点大小为15,透明度为1
    axs[0][0].scatter(x=datingDataMat[:,0], y=datingDataMat[:,1], color=LabelsColors,s=15, alpha=1)
    #设置标题,x轴label,y轴label
    axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比',FontProperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数',FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占',FontProperties=font)
    plt.setp(axs0_title_text, size=13, weight='bold', color='red') 
    plt.setp(axs0_xlabel_text, size=11, weight='bold', color='black') 
    plt.setp(axs0_ylabel_text, size=11, weight='bold', color='black')
 
    #画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为1
    axs[0][1].scatter(x=datingDataMat[:,0], y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=1)
    #设置标题,x轴label,y轴label
    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰激淋公升数',FontProperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数',FontProperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激淋公升数',FontProperties=font)
    plt.setp(axs1_title_text, size=13, weight='bold', color='red') 
    plt.setp(axs1_xlabel_text, size=11, weight='bold', color='black') 
    plt.setp(axs1_ylabel_text, size=11, weight='bold', color='black')
 
    #画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为1
    axs[1][0].scatter(x=datingDataMat[:,1], y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=1)
    #设置标题,x轴label,y轴label
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数',FontProperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比',FontProperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数',FontProperties=font)
    plt.setp(axs2_title_text, size=13, weight='bold', color='red') 
    plt.setp(axs2_xlabel_text, size=11, weight='bold', color='black') 
    plt.setp(axs2_ylabel_text, size=11, weight='bold', color='black')
    #设置图例
    didntLike = mlines.Line2D([], [], color='black', marker='.',
                      markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.',
                      markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.',
                      markersize=6, label='largeDoses')
    #添加图例
    axs[0][0].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[0][1].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[1][0].legend(handles=[didntLike,smallDoses,largeDoses])
    #显示图片
    plt.show()


#归一化数据:是数据按比例处于0`1
#tile:拓展向量,将一个一维数组拓展成多维
def autoNorm(dataSet):
    #获得数据的最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    #最大值和最小值的范围
    ranges = maxVals - minVals
    #shape(dataSet)返回dataSet的矩阵行列数
    normDataSet = np.zeros(np.shape(dataSet))
    #返回dataSet的行数
    m = dataSet.shape[0]
    #原始值减去最小值 
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    #除以最大和最小值的差,得到归一化数据
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    #返回归一化数据结果,数据范围,最小值
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix('ch2\datingTestSet.txt')
    normMat,ranges,minvals = autoNorm(datingDataMat)
    m=normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],5)
        print("分类结果:%d\t真实类别:%d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("错误率:%f%%" %(errorCount/float(numTestVecs)*100))

def classifyPerson():
    #输出结果
    resultList = ['讨厌','有些喜欢','非常喜欢']
    #三维特征用户输入
    precentTats = float(input("玩视频游戏所耗时间百分比:"))
    ffMiles = float(input("每年获得的飞行常客里程数:"))
    iceCream = float(input("每周消费的冰激淋公升数:"))
    #打开的文件名
    filename = "datingTestSet.txt"
    #打开并处理数据
    datingDataMat, datingLabels = file2matrix(filename)
    #训练集归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)
    #生成NumPy数组,测试集
    inArr = np.array([ffMiles, precentTats, iceCream])
    #测试集归一化
    norminArr = (inArr - minVals) / ranges
    #返回分类结果
    classifierResult = classify0(norminArr, normMat, datingLabels, 3)
    #打印结果
    print("你可能%s这个人" % (resultList[classifierResult-1]))


if __name__ == '__main__':
    classifyPerson()
