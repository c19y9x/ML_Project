from math import log
import operator

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet: #the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2) #log base 2
    return shannonEnt
    
# 代码功能：划分数据集
def splitDataSet(dataSet,axis,value): #传入三个参数第一个参数是我们的数据集，是一个链表形式的数据集；第二个参数是我们的要依据某个特征来划分数据集
    retDataSet = [] #由于参数的链表dataSet我们拿到的是它的地址，也就是引用，直接在链表上操作会改变它的数值，所以我们新建一格链表来做操作

    for featVec in dataSet:
        if featVec[axis] == value: #如果某个特征和我们指定的特征值相等
        #除去这个特征然后创建一个子特征
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:])
            #将满足条件的样本并且经过切割后的样本都加入到我们新建立的样本中
            retDataSet.append(reduceFeatVec)

    return retDataSet
    
def choosebestfeaturetosplit(dataset):   #就算出信息增益之后选取信息增益值最高的特征作为下一次分类的标准
    numfeatures=len(dataset[0])-1     #计算特征数量，列表【0】表示列的数量，-1是减去最后的类别特征
    baseentropy=calcShannonEnt(dataset)   #计算数据集的信息熵
    bestinfogain=0.0;bestfeature=-1
    for i in range(numfeatures):  
        featlist=[example[i] for example in dataset]
        uniquevals=set(featlist)   #确定某一特征下所有可能的取值
        newentropy=0.0
        for value in uniquevals:
            subdataset=splitDataSet(dataset,i,value)#抽取在该特征的每个取值下其他特征的值组成新的子数据集
            prob=len(subdataset)/float(len(dataset))#计算该特征下的每一个取值对应的概率（或者说所占的比重）
            newentropy +=prob*calcShannonEnt(subdataset)#计算该特征下每一个取值的子数据集的信息熵
        infogain=baseentropy-newentropy   #计算每个特征的信息增益
      #  print("第%d个特征是的取值是%s，对应的信息增益值是%f"%((i+1),uniquevals,infogain))
        if(infogain>bestinfogain):
            bestinfogain=infogain
            bestfeature=i
   # print("第%d个特征的信息增益最大，所以选择它作为划分的依据，其特征的取值为%s,对应的信息增益值是%f"%((i+1),uniquevals,infogain))
    return bestfeature

def majoritycnt(classlist):
#针对所有特征都用完，但是最后一个特征中类别还是存在很大差异，
#比如西瓜颜色为青绿的情况下同时存在好瓜和坏瓜，无法进行划分，此时选取该类别中最多的类

#作为划分的返回值，majoritycnt的作用就是找到类别最多的一个作为返回值
    classcount={}#创建字典
    for vote in classlist:
        if vote not in classcount.keys():
            classcount[vote]=0   #如果现阶段的字典中缺少这一类的特征，创建到字典中并令其值为0
        classcount[vote] +=1 #循环一次，在对应的字典索引vote的数量上加一
        sortedclasscount=sorted(classcount.items(),
        key=operator.itemgetter(1),reverse=True)#operator.itemgetter(1)是抓取其中第2个数据的值
        #利用sorted方法对class count进行排序，并且以key=operator.itemgetter(1)作为排序依据降序排序因为用了（reverse=True）,3.0以上的版本不再有iteritems而是items
    return sortedclasscount[0][0]

def createtree(dataset,labels):
    classlist=[example[-1] for example in dataset]   #提取dataset中的最后一栏——种类标签
    if classlist.count(classlist[0])==len(classlist): #计算classlist[0]出现的次数,如果相等，说明都是属于一类，不用继续往下划分
        return classlist[0]
    if len(dataset[0])==1:   #看还剩下多少个属性，如果只有一个属性，但是类别标签又多个，就直接用majoritycnt方法进行整理  选取类别最多的作为返回值
        return majoritycnt(classlist)
    bestfeat=choosebestfeaturetosplit(dataset)#选取信息增益最大的特征作为下一次分类的依据
    bestfeatlabel=labels[bestfeat]   #选取特征对应的标签
    mytree={bestfeatlabel:{}}   #创建tree字典，紧跟现阶段最优特征，下一个特征位于第二个大括号内，循环递归
    del(labels[bestfeat])   #使用过的特征从中删除
    featvalues=[example[bestfeat] for example in dataset]  #特征值对应的该栏数据
    uniquevals=set(featvalues)   #找到featvalues所包含的所有元素，同名元素算一个
    for value in uniquevals:
        sublabels=labels[:]  #子标签的意思是循环一次之后会从中删除用过的标签 ，剩下的就是子标签了
        mytree[bestfeatlabel][value]=createtree(splitDataSet(dataset,bestfeat,value),sublabels)   #循环递归生成树
    return mytree