# -*- coding:utf-8 -*-
"""
  @author:ly
  @file: kNNDigits.py
  @time: 2018/6/2211:36
  @version: v1.0
  @Desc: 使用sklearn实现kNN
"""

import numpy as np
import PIL.Image as Image
from os import listdir,remove
import matplotlib.pylab as plt
from sklearn.neighbors import KNeighborsClassifier as kNN

"""
    函数说明：将手写数字转换成二进制文件
    :parameter
        imagePath - 图片路径
    :returns
        digitFile - 二进制文件
    
"""

def imageToArray(imagePath):

    # 打开图片
    image = Image.open(imagePath).convert('RGBA')

    #得到图片像素值
    raw_data = image.load()

    #将其降噪并转化为黑白两色
    for y in range(image.size[1]):
        for x in range(image.size[0]):
            if raw_data[x,y][0] < 90:
                raw_data[x,y] = (0,0,0,255)
    for y in range(image.size[1]):
        for x in range(image.size[0]):
            if raw_data[x,y][1] < 136:
                raw_data[x,y] = (0,0,0,255)
    for y in range(image.size[1]):
        for x in range(image.size[0]):
            if raw_data[x,y][2] > 0:
                raw_data[x,y] = (255,255,255,255)

    #设置为32*32的大小
    image = image.resize((32,32),Image.LANCZOS)

    #进行保存
    image.save('data/middle.png')

    #得到像素数组，为(32,32,4)
    array = plt.array(image)

    #按照公式将其转换为01，公式：0.299*R+0.587*G+0.114*B
    gray_array = np.zeros((32,32))

    #行数
    for i in range(array.shape[0]):
        #列数
        for j in range(array.shape[1]):
            #计算灰度，若为255则白色，数值越小越接近黑色
            gray = 0.299*array[i][j][0]+0.587*array[i][j][1]+0.114*array[i][j][2]

            #设置一个阀值，记为0
            if gray == 255:
                gray_array[i][j] = 0
            else:
                gray_array[i][j] = 1

    #得到对应名称的txt文件
    filename = imagePath.split("/")[1]
    name01 = filename.split('.')[0]
    digitFile = 'data/'+name01 + '.txt'
    remove('data/middle.png')
    # 保存到文件中
    np.savetxt(digitFile, gray_array, fmt='%d', delimiter='')
    return  digitFile

"""
    函数说明：将32*32的二进制图像转换成1*1024向量
    :parameter
        filename - 文件名
    :returns
        returnVect - 返回的二进制图像的1*1024向量

"""

def imgToVector(filename):

    #创建1*1024的零向量
    returnVect = np.zeros((1,1024))
    #打开文件
    with open(filename) as fr:
        #按行读取
        for i in range(32):
            lineStr = fr.readline().strip()
            #每一行的前32个元素依次添加到returnVect中
            for j in range(32):
                returnVect[0,32*i+j] = int(lineStr[j])
    #返回转换后的1*1024向量
    return returnVect

"""
    函数说明：手写数字分器模型
    :parameter
        无
    :returns
        neigh - 训练分类器模型

"""
def handwritingClassTrain():

    # 测试集的lables
    hwLabels = []
    # 返回trainingDigits目录下的文件名
    trainingDigits = listdir('data/trainingDigits')
    # 返回文件夹下的文件个数
    count = len(trainingDigits)
    #初始化训练的Mat矩阵，测试集
    trainingMat = np.zeros((count,1024))
    #从文件名中解析出训练集的类别
    for i in range(count):
        #获得文件的名字
        filenameStr = trainingDigits[i]
        #获得分类的数字
        classNumber = int(filenameStr.split('_')[0])
        #将获得的类别添加到hwLabels中
        hwLabels.append(classNumber)
        #将每一个文件的1*1024数据存储到trainingMat矩阵中
        trainingMat[i,:] = imgToVector('data/trainingDigits/%s' %(filenameStr))
    #构建kNN分类器
    neigh = kNN(n_neighbors=5,algorithm='auto')
    #拟合模型，trainingMat为测试矩阵，hwLabels为对应的标签
    neigh.fit(trainingMat,hwLabels)
    return neigh

"""
    函数说明：手写分类器模型测试
    :parameter
        无
    :returns
        无

"""

def handwritingClassTest():
    neigh = handwritingClassTrain()
    # 返回testDigits目录下的文件列表
    testDigits = listdir('data/testDigits')
    # 错误检测计数
    errercount = 0.0
    # 测试数据数量
    testCount = len(testDigits)
    # 从文件中解析出测试集的类别并进行分类测试
    for i in range(testCount):
        # 获得文件名字
        filenameStr = testDigits[i]
        # 获得分类的数字
        classNumber = int(filenameStr.split('_')[0])
        # 获得测试集的1*1024向量,用于训练
        vectorUnderTest = imgToVector('data/testDigits/%s' % (filenameStr))
        # 获得预测结果
        classifierResult = neigh.predict(vectorUnderTest)
        print("分类返回结果为%d\t真实结果为%d" % (classifierResult, classNumber))
        if (classifierResult != classNumber):
            errercount += 1.0
    print("一共%d个测试数据，总共错了%d个数据\n错误率为%f%%" % (testCount, errercount, errercount / testCount * 100))


"""
    函数说明：使用真实手写数字获取数字
    :parameter
        imagePath - 真实手写数字图片
    :returns
        digit - 识别出的数字

"""

def handwritingTrust(imagePath):
    neigh = handwritingClassTrain()
    digitFile = imageToArray(imagePath)
    vectorTrust = imgToVector(digitFile)
    digit = neigh.predict(vectorTrust)
    print("程序识别出的数字是%d"%digit)


if __name__ == '__main__':
    imagePath = "data/4.png"
    handwritingTrust(imagePath)