# -*- coding:utf-8 -*-
"""
  @author:ly
  @file: decisionTree.py
  @time: 2018/6/2510:37
  @version: v1.0
  @Dec: 实现决策树
"""

from math import log
import operator
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pickle
import os
import time
from sklearn import tree


"""
函数说明：计算给定数据集的经验熵（香农熵）

:parameter
    dataSet - 数据集
:returns
    shannonEnt - 经验熵（香农熵）

"""


def calcShannonEnt(dataSet):
    # 返回数据集的行数
    numEntires = len(dataSet)

    # 保存每个标签（label）出现次数的字典
    labelCounts = {}

    # 对每组特征向量进行统计
    for featVec in dataSet:
        # 当前标签（label）信息
        currentLabel = featVec[-1]
        # 如果标签没有放入到统计次数的字典，添加进去
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        # label计数
        labelCounts[currentLabel] += 1
    # 初始设置经验熵
    shannonEnt = 0.0
    for key in labelCounts:
        # 选择该标签的概率
        prod = float(labelCounts[key]) / numEntires
        # 利用公式计算
        shannonEnt -= prod * log(prod, 2)
    # 返回经验熵
    return shannonEnt


"""
函数说明：创建测试数据集

:parameter
    无

:returns
    dataSet - 数据集
    labels - 特征标签

"""

def createDataSet():

    # 数据集
    dataSet = [[0, 0, 0, 0, 'no'],
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    labels = ['年龄','有工作','有自己的房子','信贷情况']
    return dataSet,labels

"""
函数说明：按照给定特征划分数据集

:parameter
    dataSet - 待划分的数据集
    axis - 划分数据集的特征
    value - 需要返回的特征的值

:returns
    无

"""

def splitDataSet(dataSet,axis,value):

    # 创建返回的数据集列表
    retDataSet = []
    # 遍历数据集
    for featVec in dataSet:
        if featVec[axis] == value:
            # 去掉axis特征
            reducedFeatVec = featVec[:axis]
            # 将符合条件的添加到返回的数据集
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    # 返回划分后的数据集
    return retDataSet

"""
函数说明：选择最优特征

:parameter
    dataSet - 数据集

:returns
    bestFeature - 信息增益最大的（最优）特征的索引值

"""
def chooseBestFeatureToSplit(dataSet):

    # 特征数量
    numFeatures = len(dataSet[0]) - 1
    # 计算数据集的香农熵
    baseEntropy = calcShannonEnt(dataSet)
    # 信息增益
    bestInfoGain = 0.0
    # 最优特征的索引值
    bestFeature = -1
    # 遍历所有特征
    for i in range(numFeatures):
        # 获取dataSet的第i个特征
        featlist = [example[i] for example in dataSet]
        # 创建set集合，去重，元素不可重复
        uniquevals = set(featlist)
        # 经验条件熵
        newEntropy = 0.0
        # 计算信息增益
        for value in uniquevals:
            # subDataSet划分后的子集
            subDataSet = splitDataSet(dataSet,i,value)
            # 计算子集的概率
            prob  = len(subDataSet) / float(len(dataSet))
            # 根据公式计算经验条件熵
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 信息增益
        infoGain = baseEntropy - newEntropy
        # 打印每个特征的信息增益
        print("第%d个特征的增益为%.3f" %(i,infoGain))
        if (infoGain > bestInfoGain):
            # 更新信息增益，找到最大信息增益
            bestInfoGain = infoGain
            # 记录最大信息增益的特征的索引值
            bestFeature = i
    # 返回信息增益最大的特征的索引值
    return bestFeature

"""
函数说明：统计classList中出现此处最多的元素（类标签）

:parameter
    classList - 类标签列表

:returns
    sortedClassCount[0][0] - 出现此处最多的元素（类标签）

"""
def majorityCnt(classList):

    classCount = {}
    # 统计classList中每个元素出现的次数
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    # 返回classList中出现次数最多的元素
    return sortedClassCount[0][0]

"""
函数说明：创建决策树

:parameter
    dataSet - 训练数据集
    labels - 分类属性标签
    featLabels - 存储选择的最优特征标签
    
:returns
    myTree - 决策树

"""

def createTree(dataSet,labels,featLabels):

    # 获取分类标签（是否放贷：yes or no）
    classList = [example[-1] for example in dataSet]

    # 如果类别完全相同则停止继续划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    # 遍历完所有特征时返回出现次数最多的类标签
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    # 选择最优特征
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 最优特征的标签
    bestFeatLabel = labels[bestFeat]
    featLabels.append(bestFeatLabel)
    # 根据最优特征的标签生成树
    myTree = {bestFeatLabel:{}}
    # 删除已经使用特征标签
    del(labels[bestFeat])
    # 得到训练集中所有最优特征的属性值
    featValues = [example[bestFeat] for example in dataSet]
    #去掉重复的属性值
    uniqueVlas = set(featValues)
    # 遍历特征，创建决策树
    for value in uniqueVlas:
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),labels,featLabels)
    # 返回决策树
    return myTree

"""
函数说明：获取决策树叶子节点的数目

:parameter
    myTree - 决策树

:returns
    numLeafs - 决策树的叶子节点数目

"""

def getNumLeafs(myTree):

    # 初始化叶子节点数目
    numLeafs = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

"""
函数说明：获取决策树层数

:parameter
    myTree - 决策树

:returns
    maxDepth - 决策树的层数

"""
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

"""
函数说明：绘制结点

:parameter
    nodeTxt - 结点名
    centerPt - 文本位置
    parentPt - 标注箭头的位置
    nodeType - 结点格式

:returns
    无

"""
def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    arrow_args = dict(arrowstyle="<-")
    # 设置汉字格式
    font = FontProperties(fname="C:\Windows\Fonts\simsun.ttc", size=14)
    # 绘制结点
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',xytext=centerPt,
                            textcoords='axes fraction',va='center',ha='center',bbox=nodeType,
                            arrowprops=arrow_args,FontProperties=font)


"""
函数说明：标注有向边属性值

:parameter
    cntrPt、parentPt - 用于计算标注位置
    txtString - 标注的内容
    
:returns
    无

"""
def plotMidText(cntrPt,parentPt,txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid,yMid,txtString,va='center',ha='center',rotation=30)


"""
函数说明：绘制决策树

:parameter
    myTree - 决策树（字典）
    parentPt - 标注的内容
    nodeTxt - 结点名

:returns
    无

"""
def plotTree(myTree,parentPt,nodeTxt):
    decisionNode = dict(boxstyle="sawtooth",fc="0.8")
    leafNode = dict(boxstyle="round4",fc="0.8")
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    # 跟结点
    firstStr = next(iter(myTree))
    # 中心位置
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff)
    plotMidText(cntrPt,parentPt,nodeTxt)
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    secondDict = myTree[firstStr]
    # y偏移
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 /plotTree.totalW
            plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrPt,leafNode)
            plotMidText((plotTree.xOff,plotTree.yOff),cntrPt,str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD



"""
函数说明：创建绘制面板

:parameter
    inTree - 决策树（字典）

:returns
    无

"""
def createPlot(inTree):
    # 创建fig
    fig = plt.figure(1,facecolor="white")
    # 清空fig
    fig.clf()
    # 去掉x,y轴
    axprops = dict(xticks=[],yticks=[])
    createPlot.ax1 = plt.subplot(111,frameon=False,**axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree,(0.5,1.0),'')
    plt.show()


"""
函数说明：使用决策树分类

:parameter
    inputTree - 已经生成的决策树
    feaLabels - 存储选择的最优特征标签
    testVec - 测试数据列表，顺序对应最优特征标签

:returns
    classLabel - 分类结果

"""
def classify(inputTree,featLabels,testVec):
    firstStr = next(iter(inputTree))  # 获取决策树结点
    secondDict = inputTree[firstStr]  # 下一个字典
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


"""
函数说明：存储决策树

:parameter
    inputTree - 已经生成的决策树
    filename - 决策树的存储文件名

:returns
    无

"""
def storeTree(inputTree,filename):
    if not os.path.exists(filename.split('/')[0]):
        os.mkdir('data')
    if not os.path.isfile(filename):
        with open(filename,'wb+') as fw:
            pickle.dump(inputTree,fw)

"""
函数说明：读取决策树

:parameter
    filename - 决策树的存储文件名

:returns
    pickle.load(fr) - 决策树字典

"""
def grabTree(filename):
    with open(filename,'rb') as fr:
        return pickle.load(fr)

if __name__ == '__main__':
    start = time.clock()
    dataSet,labels = createDataSet()
    featLabels = []
    myTree = createTree(dataSet,labels,featLabels)
    storeTree(myTree,'data/classifierStorage')
    myTree = grabTree('data/classifierStorage')
    print(myTree)
    # createPlot(myTree)
    testVec = [0,1]
    result = classify(myTree,featLabels,testVec)
    if result == 'yes':
        print("可以放贷")
    elif result == 'no':
        print("不能放贷")
    else:
        print('未知结果，不放贷')
    # clf = tree.DecisionTreeClassifier(criterion='entropy')
    # label_value = []
    # for i in range(len(dataSet)):
    #     label_value.append(dataSet[i][-1])
    #     dataSet[i] = dataSet[i][:-1]
    #
    # print(dataSet,label_value)
    # clf = clf.fit(dataSet,label_value)
    # print(clf.predict([[0,0,0,2]]))
    end = time.clock()
    print("final is in",end-start)

