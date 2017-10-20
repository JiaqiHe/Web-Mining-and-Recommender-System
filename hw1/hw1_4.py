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

def problem1_4():
    m = {} #dictionary to store numbers of reviews for each style of beer
    for d in data:
        if d['beer/style'] in m:
            m[d['beer/style']] += 1
        else:
            m[d['beer/style']] = 1

    n = {} #store >=50 reviews beers
    excluded = set([]) #store those beers which get excluded
    for key,value in m.items():
        if value >= 50:
            n[key] = len(n)+1
        else:
            excluded.add(key)

    print ('There are ', len(n), 'beers which have more than 50 reviews. They are:')
    print (n)

    X=[]
    Y=[]
    for d in data:
        if d['beer/style'] not in excluded:
            cur = [0] * (len(n)+1)
            cur[0] = 1
            cur[n[d['beer/style']]] = 1
            X.append(cur)
            Y.append(d['review/taste'])
        else:
            cur = [0] * (len(n)+1)
            cur[0] = 1
            X.append(cur)
            Y.append(d['review/taste'])

    X1 = X[:int(len(X)/2)]
    X2 = X[int(len(X)/2):]
    Y1 = Y[:int(len(Y)/2)]
    Y2 = Y[int(len(Y)/2):]
    len1 = len(X1)
    len2 = len(X2)
    theta,residuals,rank,s = numpy.linalg.lstsq(X1, Y1)
    print (theta)
    print ('MSE of training part is: ', residuals/len1)

    X2 = numpy.matrix(X2)
    Y2 = numpy.matrix(Y2)
    A = numpy.dot(theta, X2.T)
    mse = numpy.square(A - Y2).mean()
    print ('MSE of testing part is: ', mse)

problem1_4()
