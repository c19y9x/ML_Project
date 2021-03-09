import numpy as np
import kNN

datingDataMat,datingLabels = kNN.file2matrix('datingTestSet.txt')
print(datingDataMat.min(1))