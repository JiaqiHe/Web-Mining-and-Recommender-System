import numpy
import urllib
import scipy.optimize
import random
from urllib import request

# def parseData(fname):
#   i = 1
#   for l in urllib.request.urlopen(fname):
#     if i < 21:
#       yield eval(l)
#       i = i+1
#     else:
#       break

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
X = numpy.matrix(X)
y = numpy.matrix(y)
print (numpy.linalg.inv(X.T * X) * X.T * y.T)
# theta,residuals,rank,s = numpy.linalg.lstsq(X, y)
# print (theta)
