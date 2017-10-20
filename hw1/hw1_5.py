import numpy
import urllib
import scipy.optimize
import random
from urllib import request
from sklearn import svm

def parseData(fname):
  for l in urllib.request.urlopen(fname):
    yield eval(l)

print ("Reading data...")
data = list(parseData("http://jmcauley.ucsd.edu/cse255/data/beer/beer_50000.json"))
print ("done")

X = [[b['beer/ABV'],b['review/taste']] for b in data]
y = ["American IPA" in b['beer/style'] for b in data]

length = len(X)
X_train = X[:int(length/2)]
y_train = y[:int(length/2)]
X_test = X[int(length/2):]
y_test = y[int(length/2):]

# Create a support vector classifier object, with regularization parameter C = 1000
clf = svm.SVC(C=1000, kernel='linear')
clf.fit(X_train, y_train)

train_predictions = clf.predict(X_train)
test_predictions = clf.predict(X_test)

match_train = [(x==y) for x,y in zip(train_predictions, y_train)]
match_test  = [(x==y) for x,y in zip(test_predictions, y_test)]

print ("The accuracy on the train data is", sum(match_train)*1.0/len(match_train))
print ("The accuracy on the test data is", sum(match_test)*1.0/len(match_test))
