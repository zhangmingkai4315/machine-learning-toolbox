# -*- coding:utf-8 -*-
from numpy import *
import operator
import os
from os import listdir
import matplotlib
import matplotlib.pyplot as plt

project_folder = os.path.dirname(os.path.realpath(__file__))


def classify0(inX, dataSet, labels, k):
  dataSetSize = dataSet.shape[0]
  diffMat = tile(inX, (dataSetSize, 1)) - dataSet
  sqDiffMat = diffMat**2
  sqDistances = sqDiffMat.sum(axis=1)
  distances = sqDistances ** 0.5

  sortedDistIndicies = distances.argsort()
  classCount = {}

  for i in range(k):
    voteIlabel = labels[sortedDistIndicies[i]]
    classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

  sortedClassCount = sorted(classCount.iteritems(),
                            key=operator.itemgetter(1), reverse=True)
  return sortedClassCount[0][0]


def file2Matrix(filename):
  f = open(filename)
  arrayOfLines = f.readlines()
  numOfLines = len(arrayOfLines)
  returnMat = zeros((numOfLines, 3))
  classLabelVector = []
  index = 0
  for line in arrayOfLines:
    line = line.strip()
    listFromLine = line.split('\t')
    returnMat[index, :] = listFromLine[0:3]
    classLabelVector.append(int(listFromLine[-1]))
    index += 1
  return returnMat, classLabelVector


def autoNorm(dataSet):
  minVals = dataSet.min(0)
  maxVals = dataSet.max(0)
  ranges = maxVals - minVals
  normDataset = zeros(shape(dataSet))
  m = dataSet.shape[0]
  normDataset = dataSet - tile(minVals, (m, 1))
  normDataset = normDataset * 1.0 / tile(ranges, (m, 1))
  return normDataset, ranges, minVals


def img2Vector(filename):
  returnVect = zeros((1, 1024))
  f = open(filename)
  for i in range(32):
    lineStr = f.readline()
    for j in range(32):
      returnVect[0, 32 * i + j] = int(lineStr[j])
  return returnVect


def handwritingClassTest():
  hwLabels = []
  trainingFileList = listdir('trainingDigits')
  m = len(trainingFileList)
  trainingMat = zeros((m, 1024))
  for i in range(m):
    fileNameStr = trainingFileList[i]
    fileStr = fileNameStr.split('.')[0]
    classNumStr = int(fileStr.split('_')[0])
    hwLabels.append(classNumStr)
    trainingMat[i, :] = img2Vector('trainingDigits/%s' % fileNameStr)

  testFileList = listdir('testDigits')
  errorCount = 0.0

  mtest = len(testFileList)
  for i in range(mtest):
    fileNameStr = testFileList[i]
    fileStr = fileNameStr.split('.')[0]
    classNumStr = int(fileStr.split('_')[0])

    vectorUnderTest = img2Vector('testDigits/%s' % fileNameStr)

    classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)

    if classifierResult != classNumStr:
      errorCount += 1.0
    print "the classifier came back with :%d,the real answe is: %d" % (classifierResult, classNumStr)
  print "the error rate is :%f" % (errorCount / float(mtest))


def main():
  handwritingClassTest()


if __name__ == '__main__':
  main()
