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

data = list(parseData("http://jmcauley.ucsd.edu/cse255/data/beer/beer_50000.json"))

m = {} #dictionary to store numbers of reviews for each style of beer
average = {} #dictionary to store average values of reviews
for d in data:
    if d['beer/style'] in m:
        m[d['beer/style']] += 1
        average[d['beer/style']] += d['review/taste']
    else:
        m[d['beer/style']] = 1
        average[d['beer/style']] = d['review/taste']

for style in average:
    average[style] = average[style]/m[style]

print (m)
print (average)
