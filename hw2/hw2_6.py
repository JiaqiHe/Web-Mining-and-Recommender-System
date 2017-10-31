import numpy
from urllib.request import urlopen
import scipy.optimize
import random
from math import exp
from math import log
from sklearn.decomposition import PCA
import re

def parseData(fname):
  for l in urlopen(fname):
    yield eval(l)

print("Reading data...")
data = list(parseData("http://jmcauley.ucsd.edu/cse190/data/beer/beer_50000.json"))
print("done")

def feature(datum):
  rev = datum['review/text'].lower()
  rev = rev.replace('\t', '')
  rev = re.sub(r'[^\w\s]','',rev)
  # table = str.maketrans({key: None for key in string.punctuation})
  target = ['lactic','tart','sour','citric','sweet','acid','hop','fruit','salt','spicy']
  feat = [0] * 10
  for word in rev.split():
    #   word = word.translate(table)
      for i in range(10):
          if word == target[i]:
              feat[i] += 1;
  return feat

X = [feature(d) for d in data]
y = ["American IPA" in b['beer/style'] for b in data]

X_train = X[:int(len(X)/3)]
y_train = y[:int(len(X)/3)]
X_validate = X[int(len(X)/3):2*int(len(X)/3)]
y_validate = y[int(len(X)/3):2*int(len(X)/3)]
X_test = X[2*int(len(X)/3):]
y_test = y[2*int(len(X)/3):]

pca = PCA(n_components = 10)
pca.fit(X_train)
eigen_matrix = pca.components_
print(len(X_train) * numpy.sum(pca.explained_variance_[2:]))
