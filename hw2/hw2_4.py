import numpy
from urllib.request import urlopen
import scipy.optimize
import random
from math import exp
from math import log
import string
import re

def parseData(fname):
    with open(fname,'r') as fp:
        for l in fp.readlines():
            yield eval(l)

print("Reading data...")
data = list(parseData("beer_50000.json"))
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
  feat.insert(0,1)
  return feat

X = [feature(d) for d in data]
y = [d['beer/ABV'] >= 6.5 for d in data]


countnumber = [0] *11
for i in range(int(len(X)/3)):
    for j in range(11):
        countnumber[j] += X[i][j]
print(countnumber)
#
#
def inner(x,y):
  return sum([x[i]*y[i] for i in range(len(x))])

def sigmoid(x):
  return 1.0 / (1 + exp(-x))

##################################################
# Logistic regression by gradient ascent         #
##################################################

# NEGATIVE Log-likelihood
def f(theta, X, y, lam):
  t = sum(y) #the number of positive samples
  f = len(y)-t #the number of negative samples
  loglikelihood = 0
  for i in range(len(X)):
    logit = inner(X[i], theta)
    if not y[i]:
      loglikelihood -= weight_neg * logit
      loglikelihood -= weight_neg * log(1 + exp(-logit))
    else:
      loglikelihood -= weight_pos * log(1 + exp(-logit))
  for k in range(len(theta)):
    loglikelihood -= lam * theta[k]*theta[k]
  # for debugging
  # print("ll =" + str(loglikelihood))
  return -loglikelihood

# NEGATIVE Derivative of log-likelihood
def fprime(theta, X, y, lam):
  dl = [0]*len(theta)
  for i in range(len(X)):
    logit = inner(X[i], theta)
    for k in range(len(theta)):
      if not y[i]:
        dl[k] -= weight_neg * X[i][k]
        dl[k] += weight_neg * X[i][k] * (1 - sigmoid(logit))
      else:
        dl[k] += weight_pos * X[i][k] * (1 - sigmoid(logit))
  for k in range(len(theta)):
    dl[k] -= lam*2*theta[k]
  return numpy.array([-x for x in dl])

X_train = X[:int(len(X)/3)]
y_train = y[:int(len(X)/3)]
X_validate = X[int(len(X)/3):2*int(len(X)/3)]
y_validate = y[int(len(X)/3):2*int(len(X)/3)]
X_test = X[2*int(len(X)/3):]
y_test = y[2*int(len(X)/3):]

train_total = len(y_train)
positive_total = sum(y_train)
negative_total = train_total - positive_total
weight_pos = train_total/(2.0*positive_total)
weight_neg = train_total/(2.0*negative_total)
##################################################
# Train                                          #
##################################################

def train(lam):
  theta,_,_ = scipy.optimize.fmin_l_bfgs_b(f, [0]*len(X[0]), fprime, pgtol = 10, args = (X_train, y_train, lam))
  return theta

##################################################
# Predict                                        #
##################################################

def performance(theta):
  scores = [inner(theta,x) for x in X_test]
  predictions = [s > 0 for s in scores]
  correct = [(a==b) for (a,b) in zip(predictions,y_test)]
  TP = 0
  TN = 0
  FP = 0
  FN = 0
  for (a,b) in zip(predictions, y_test):
      if a==True:
          if b == True:
              TP += 1
          else:
              FP += 1
      else:
          if b == True:
              FN += 1
          else:
              TN += 1
  print("# of true positive is",TP)
  print("# of true negative is",TN)
  print("# of false positive is",FP)
  print("# of false negative is",FN)
  BER = (FP/(FP+TN)+FN/(FN+TP))/2.0
  print("Balanced Error Rate is", BER)
  acc = sum(correct) * 1.0 / len(correct)
  return acc

##################################################
# Validation pipeline                            #
##################################################

lam = 1.0

theta = train(lam)
print("theta is equal to", theta)
acc = performance(theta)
print("lambda = " + str(lam) + ":\taccuracy=" + str(acc))
