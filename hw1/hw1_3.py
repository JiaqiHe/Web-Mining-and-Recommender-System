import numpy
import urllib
import scipy.optimize
import random
from urllib import request

def parseData(fname):
  for l in urllib.request.urlopen(fname):
    yield eval(l)

print ("Reading data...")
data = list(parseData("http://jmcauley.ucsd.edu/cse255/data/beer/beer_50000.json"))
print ("done")

X = []
for d in data:
    if d['beer/style'] == 'American IPA':
        X.append([1, 1])
    else:
        X.append([1, 0])

y = [d['review/taste'] for d in data]
X1 = X[:int(len(X)/2)]
X2 = X[int(len(X)/2):]
y1 = y[:int(len(y)/2)]
y2 = y[int(len(y)/2):]
theta,residuals,rank,s = numpy.linalg.lstsq(X1, y1)
print ('MSE of training part is: ', residuals[0]/len(y1))

X2 = numpy.matrix(X2)
y2 = numpy.matrix(y2)
# array1 = numpy.subtract(numpy.dot(theta, X2.T),y2)
# array2 = numpy.array(array1)**2
# res = numpy.sum(array2)
# res /= int(len(X)/2)
# print ('MSE of testing part is: ', res)
A = numpy.dot(theta, X2.T)
mse = numpy.square(A - y2).mean()
print ('MSE of testing part is: ', mse)
