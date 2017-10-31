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
matrix_w = numpy.hstack((eigen_matrix[0].reshape(10,1),
                      eigen_matrix[1].reshape(10,1)))
Y = numpy.dot(X, matrix_w)
is_American_IPA = []
not_American_IPA = []
for i in range(len(y)):
    if y[i]==True:
        is_American_IPA.append(Y[i])
    else:
        not_American_IPA.append(Y[i])
x_list_1 = [r[0] for r in is_American_IPA]
y_list_1 = [r[1] for r in is_American_IPA]
x_list_2 = [r[0] for r in not_American_IPA]
y_list_2 = [r[1] for r in not_American_IPA]
# numpy.savetxt('angle1.txt',Y,fmt='%f',delimiter=' ',newline='\r\n')
import matplotlib.pyplot as plt
fig = plt.figure(0)
plt.xlabel('x')
plt.ylabel('y')
plt.title('figure')
# c='red'定义为红色，alpha是透明度，marker是画的样式
plt.scatter(x_list_2, y_list_2, c='blue', alpha=1, marker='+', label='Non-American IPA')
plt.scatter(x_list_1, y_list_1, c='red', alpha=1, marker='+', label='American IPA')
plt.grid(True)
plt.legend(loc='best')
plt.show()
