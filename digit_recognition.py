#! /usr/bin/python

from numpy import *
from os import listdir
import operator

def knn(invec, dataset, labelvec, k):
    datasetSize = dataset.shape[0]
    diffMat = tile(invec, (datasetSize,1)) - dataset
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()     
    classCount={}          
    for i in range(k):
        voteIlabel = labelvec[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def img_to_vector(filename):
    vector = zeros((1,1024))
    img = open(filename)
    for i in range(32):
        line = img.readline()
        for j in range(32):
            vector[0,32*i+j] = int(line[j])
    return vector

def file_to_data(path):
    filelist = listdir(path)
    listlen = len(filelist)
    data = zeros((listlen,1024))
    label = []
    for i in range(listlen):
        filename = (filelist[i]).split('.')[0]
        classnum = int(filename.split('_')[0])
        label.append(classnum)
        data[i,:] = img_to_vector(path+'/%s' %(filelist[i]))
    return data, label

dataset, labelvec = file_to_data('trainingDigits')
invec = img_to_vector('testDigits/4_25.txt')
result = knn(invec, dataset, labelvec, 3)

print(result)
